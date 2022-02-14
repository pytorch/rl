from argparse import ArgumentParser
from numbers import Number
from typing import Optional, Iterable

import torch
from torch import nn

from torchrl.data import TensorSpec, UnboundedContinuousTensorSpec, DEVICE_TYPING
from torchrl.envs.common import Specs, _EnvClass
from torchrl.modules import ActorValueOperator, NoisyLinear
from torchrl.modules.distributions import (
    Delta,
    TanhNormal,
    TanhDelta,
    OneHotCategorical,
    TruncatedNormal,
)
from torchrl.modules.models.models import (
    DuelingCnnDQNet,
    DdpgCnnActor,
    DdpgCnnQNet,
    DdpgMlpQNet,
    DdpgMlpActor,
    MLP,
    ConvNet, LSTMNet,
)
from torchrl.modules.probabilistic_operators import (
    QValueActor,
    DistributionalQValueActor,
    Actor,
    ProbabilisticOperator,
)
from torchrl.modules.probabilistic_operators.actors import (
    ValueOperator,
    ActorCriticWrapper,
)

DISTRIBUTIONS = {
    "delta": Delta,
    "tanh-normal": TanhNormal,
    "categorical": OneHotCategorical,
    "tanh-delta": TanhDelta,
}

__all__ = [
    "make_dqn_actor",
    "make_ddpg_actor",
    "make_ppo_model",
    "make_sac_model",
    "parser_model_args_continuous",
    "parser_model_args_discrete",
]


def make_dqn_actor(proof_environment: _EnvClass, device, args) -> Actor:
    """
    DQN constructor helper function.

    Args:
        TODO

    Returns: A DQN policy operator.

    """
    env_specs = proof_environment.specs

    atoms = args.atoms if args.distributional else None
    linear_layer_class = torch.nn.Linear if not args.noisy else NoisyLinear
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
    in_key = "observation_pixels"

    distribution_class = DISTRIBUTIONS["delta"]
    out_features = env_specs["action_spec"].shape[0]
    actor_class = QValueActor
    actor_kwargs = {}
    if atoms:
        vmin = -3
        vmax = 3

        out_features = (atoms, out_features)
        support = torch.linspace(vmin, vmax, atoms)
        actor_class = DistributionalQValueActor
        actor_kwargs.update({"support": support})
        if issubclass(net_class, DuelingCnnDQNet):
            default_net_kwargs.update({"out_features_value": (atoms, 1)})

    net = net_class(
        out_features=out_features,
        **default_net_kwargs,
    )

    actor_kwargs.setdefault(
        "default_interaction_mode",
        "mode",
    )

    model = actor_class(
        in_keys=[in_key],
        action_spec=env_specs["action_spec"],
        mapping_operator=net,
        distribution_class=distribution_class,
        # variable_size=variable_size,
        safe=True,
        **actor_kwargs,
    ).to(device)

    # init
    with torch.no_grad():
        td = proof_environment.reset()
        model(td.to(device))
    proof_environment.close()

    return model


def make_ddpg_actor(
    proof_environment: _EnvClass,
    from_pixels: bool = True,
    noisy: bool = False,
    actor_net_kwargs: Optional[dict] = None,
    value_net_kwargs: Optional[dict] = None,
    atoms: int = 0,  # for distributional dqn
    vmin: Number = -3,
    vmax: Number = 3,
    actor_kwargs: Optional[dict] = None,
    device: DEVICE_TYPING = "cpu",
) -> torch.nn.ModuleList:
    """
    DDPG constructor helper function.

    Args:
        env_specs (Specs): specs container of the environment.
        from_pixels (bool): if True, the environment observation will be asumed to be pixels and a Conv net will be
        noisy (bool): if True, a noisy layer will be used for the value network
        needed. Otherwise, a state vector is assumed and an MLP is created.
            default: True
        actor_net_kwargs (dict, optional): kwargs for the DdpgCnnActor / DdpgMlpActor classes.
        value_net_kwargs (dict, optional): kwargs for the DdpgCnnQNet / DdpgMlpQNet classes.
        actor_kwargs (dict, optional): kwargs for the Actor class, called to instantiate the policy operator.
        value_kwargs (dict, optional): kwargs for the ProbabilisticOperator class, called to instantiate the value
            operator.

    Returns: An actor and a value operators for DDPG.

    For more details on DDPG, refer to "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING",
    https://arxiv.org/pdf/1509.02971.pdf.
    """

    actor_net_kwargs = actor_net_kwargs if actor_net_kwargs is not None else dict()
    value_net_kwargs = value_net_kwargs if value_net_kwargs is not None else dict()
    actor_kwargs = actor_kwargs if actor_kwargs is not None else dict()

    linear_layer_class = torch.nn.Linear if not noisy else NoisyLinear

    env_specs = proof_environment.specs
    out_features = env_specs["action_spec"].shape[0]
    actor_class = Actor
    if atoms:
        raise NotImplementedError
        # https://arxiv.org/pdf/1804.08617.pdf

    action_distribution_class = DISTRIBUTIONS["tanh-delta"]
    actor_kwargs.setdefault(
        "default_interaction_mode",
        "mode",
    )
    actor_net_default_kwargs = {
        "action_dim": out_features,
        "mlp_net_kwargs": {"layer_class": linear_layer_class},
    }
    actor_net_default_kwargs.update(actor_net_kwargs)
    if from_pixels:
        in_keys = ["observation_pixels"]
        actor_net = DdpgCnnActor(**actor_net_default_kwargs)

    else:
        in_keys = ["observation_vector"]
        actor_net = DdpgMlpActor(**actor_net_default_kwargs)

    actor = actor_class(
        in_keys=in_keys,
        action_spec=env_specs["action_spec"],
        mapping_operator=actor_net,
        distribution_class=action_distribution_class,
        safe=True,
        **actor_kwargs,
    )

    state_class = ValueOperator
    if from_pixels:
        value_net_default_kwargs = {
            "mlp_net_kwargs": {"layer_class": linear_layer_class}
        }
        value_net_default_kwargs.update(value_net_kwargs)

        in_keys = ["observation_pixels", "action"]
        out_keys = ["state_action_value"]
        q_net = DdpgCnnQNet(**value_net_default_kwargs)
    else:
        value_net_default_kwargs1 = {"activation_class": torch.nn.ELU}
        value_net_default_kwargs1.update(
            value_net_kwargs.get(
                "mlp_net_kwargs_net1", {"layer_class": linear_layer_class}
            )
        )
        value_net_default_kwargs2 = {
            "num_cells": [400, 300],
            "depth": 2,
            "activation_class": torch.nn.ELU,
        }
        value_net_default_kwargs2.update(
            value_net_kwargs.get(
                "mlp_net_kwargs_net2", {"layer_class": linear_layer_class}
            )
        )
        in_keys = ["observation_vector", "action"]
        out_keys = ["state_action_value"]
        q_net = DdpgMlpQNet(
            mlp_net_kwargs_net1=value_net_default_kwargs1,
            mlp_net_kwargs_net2=value_net_default_kwargs2,
        )

    value = state_class(
        in_keys=in_keys,
        out_keys=out_keys,
        mapping_operator=q_net,
    )

    module = torch.nn.ModuleList([actor, value]).to(device)

    # init
    with torch.no_grad():
        td = proof_environment.reset().to(device)
        module[0](td)
        module[1](td)
    proof_environment.close()

    return module


def make_ppo_model(
    proof_environment: _EnvClass,
    args: ArgumentParser,
    device: DEVICE_TYPING,
    in_keys_actor: Optional[Iterable[str]] = None,
    **kwargs,
) -> ActorValueOperator:
    """
    Actor-value model constructor helper function.
    Currently constructs MLP networks with immutable default arguments as described in "Proximal Policy Optimization
    Algorithms", https://arxiv.org/abs/1707.06347
    Other configurations can easily be implemented by modifying this function at will.

    Args:
        spec (TensorSpec): action_spec descriptor
        in_keys_actor (Iterable[str], optional):
            default: ["observation_vector"]
        **kwargs: kwargs to be passed to the ActorCriticOperator.

    Returns: A joined ActorCriticOperator.

    """
    proof_environment.set_seed(args.seed)
    specs = proof_environment.specs  # TODO: use env.sepcs
    action_spec = specs["action_spec"]

    if args.from_pixels and in_keys_actor is None:
        in_keys_actor = ["observation_pixels"]
        in_keys_critic = ["observation_pixels"]
    elif in_keys_actor is None:
        in_keys_actor = ["observation_vector"]
        in_keys_critic = ["observation_vector"]
    out_keys = ["action"]

    if action_spec.domain == "continuous":
        out_features = 2 * action_spec.shape[-1]
        if args.distribution == "tanh_normal":
            policy_distribution_kwargs = {
                "min": action_spec.space.minimum,
                "max": action_spec.space.maximum,
                "tanh_loc": args.tanh_normal_tanh,
                "scale_mapping": f"biased_softplus_{args.default_policy_scale}",
            }
            policy_distribution_class = TanhNormal
        elif args.distribution == "truncated_normal":
            policy_distribution_kwargs = {
                "min": action_spec.space.minimum,
                "max": action_spec.space.maximum,
                "tanh_loc": args.tanh_normal_tanh,
                "scale_mapping": f"biased_softplus_{args.default_policy_scale}",
            }
            policy_distribution_class = TruncatedNormal
    elif action_spec.domain == "discrete":
        out_features = action_spec.shape[-1]
        policy_distribution_kwargs = {}
        policy_distribution_class = OneHotCategorical
    else:
        raise NotImplementedError(
            f"actions with domain {action_spec.domain} are not supported"
        )

    if args.shared_mapping:
        if args.from_pixels:
            if in_keys_actor is None:
                in_keys_actor = ["observation_pixels"]
            common_mapping_operator = ConvNet(
                bias_last_layer=True,
                depth=None,
                num_cells=[32, 64, 64],
                kernel_sizes=[8, 4, 3],
                strides=[4, 2, 1],
            )
        else:
            if args.lstm:
                raise NotImplementedError("lstm not yet compatible with shared mapping for PPO")
            common_mapping_operator = MLP(
                num_cells=[
                    400,
                ],
                out_features=300,
                activate_last_layer=True,
            )

        policy_net = MLP(
            num_cells=[200],
            out_features=out_features,
        )

        value_net = MLP(
            num_cells=[200],
            out_features=1,
        )
        actor_value = ActorValueOperator(
            spec=action_spec,
            in_keys=in_keys_actor,
            common_mapping_operator=common_mapping_operator,
            policy_operator=policy_net,
            value_operator=value_net,
            policy_distribution_class=policy_distribution_class,
            policy_interaction_mode="random",
            policy_distribution_kwargs=policy_distribution_kwargs,
            return_log_prob=True,
            save_dist_params=True,
            **kwargs,
        ).to(device)
    else:
        if args.lstm:
            policy_net = LSTMNet(
                out_features=out_features,
                lstm_kwargs={
                    'input_size': 256,
                    'hidden_size': 256},
                mlp_kwargs={'num_cells': [256, 256], 'out_features': 256},
            )
            in_keys_actor += ["hidden0", "hidden1"]
            out_keys += ["hidden0", "hidden1", "next_hidden0", "next_hidden1"]
        else:
            policy_net = MLP(
                num_cells=[400, 300],
                out_features=out_features,
            )

        policy_po = Actor(
            action_spec,
            policy_net,
            policy_distribution_class,
            distribution_kwargs=policy_distribution_kwargs,
            in_keys=in_keys_actor,
            out_keys=out_keys,
            return_log_prob=True,
            save_dist_params=True,
        )

        value_net = MLP(
            num_cells=[400, 300],
            out_features=1,
        )
        value_po = ValueOperator(
            value_net,
            in_keys=in_keys_critic,
        )
        actor_value = ActorCriticWrapper(policy_po, value_po).to(device)

    with torch.no_grad():
        td = proof_environment.reset()
        td_device = td.to(device)
        td_device = td_device.unsqueeze(0)
        td_device = actor_value(td_device)  # for init
    return actor_value


def make_sac_model(
    proof_environment: _EnvClass,
    in_keys: Optional[Iterable[str]] = None,
    actor_net_kwargs=None,
    qvalue_net_kwargs=None,
    value_net_kwargs=None,
    double_qvalue=True,
    device: DEVICE_TYPING = "cpu",
    tanh_normal_tanh: bool = True,
    default_policy_scale: float = 1.0,
    **kwargs,
) -> nn.ModuleList:
    """
    Actor, Q-value and value model constructor helper function for SAC.
    Follows default parameters proposed in SAC original paper: https://arxiv.org/pdf/1801.01290.pdf.
    Other configurations can easily be implemented by modifying this function at will.

    Args:
        action_spec (TensorSpec): action_spec descriptor
        in_keys (Iterable[str], optional):
            default: ["observation_vector"]
        device (str, int or torch.device): device where the modules will live.
            default: "cpu"
        **kwargs: kwargs to be passed to the ActorCriticOperator.

    Returns: A joined ActorCriticOperator.

    """
    td = proof_environment.reset()
    action_spec = proof_environment.action_spec
    if actor_net_kwargs is None:
        actor_net_kwargs = {}
    if value_net_kwargs is None:
        value_net_kwargs = {}
    if qvalue_net_kwargs is None:
        qvalue_net_kwargs = {}

    if in_keys is None:
        in_keys = ["observation_vector"]

    actor_net_kwargs_default = {
        "num_cells": [256, 256],
        "out_features": 2 * action_spec.shape[-1],
        "activation_class": nn.ELU,
    }
    actor_net_kwargs_default.update(actor_net_kwargs)
    actor_net = MLP(**actor_net_kwargs_default)

    qvalue_net_kwargs_default = {
        "num_cells": [256, 256],
        "out_features": 1,
        "activation_class": nn.ELU,
    }
    qvalue_net_kwargs_default.update(qvalue_net_kwargs)
    qvalue_net = MLP(
        **qvalue_net_kwargs_default,
    )
    if double_qvalue:
        qvalue_net_bis = MLP(**qvalue_net_kwargs_default)

    value_net_kwargs_default = {
        "num_cells": [256, 256],
        "out_features": 1,
        "activation_class": nn.ELU,
    }
    value_net_kwargs_default.update(value_net_kwargs)
    value_net = MLP(
        **value_net_kwargs_default,
    )

    value_spec = UnboundedContinuousTensorSpec()

    actor = Actor(
        action_spec=action_spec,
        in_keys=in_keys,
        mapping_operator=actor_net,
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": action_spec.space.minimum,
            "max": action_spec.space.maximum,
            "tanh_loc": tanh_normal_tanh,
            "scale_mapping": f"biased_softplus_{default_policy_scale}",
        },
        default_interaction_mode="random",
    )
    qvalue = ProbabilisticOperator(
        spec=value_spec,
        in_keys=["action"] + in_keys,
        out_keys=["state_action_value"],
        mapping_operator=qvalue_net,
        distribution_class=Delta,
    )
    if double_qvalue:
        qvalue_bis = ProbabilisticOperator(
            spec=value_spec,
            in_keys=["action"] + in_keys,
            out_keys=["state_action_value"],
            mapping_operator=qvalue_net_bis,
            distribution_class=Delta,
        )
    value = ProbabilisticOperator(
        spec=value_spec,
        in_keys=in_keys,
        out_keys=["state_value"],
        mapping_operator=value_net,
        distribution_class=Delta,
    )
    if double_qvalue:
        model = nn.ModuleList([actor, qvalue, qvalue_bis, value]).to(device)
    else:
        model = nn.ModuleList([actor, qvalue, value]).to(device)

    # init nets
    td = td.to(device)
    for net in model:
        net(td)
    del td

    proof_environment.close()

    return model


def parser_model_args_continuous(
    parser: ArgumentParser, algorithm: str
) -> ArgumentParser:
    """
    To be used for DDPG, SAC
    """

    if algorithm not in ("SAC", "DDPG", "PPO"):
        raise NotImplementedError(f"Unknown algorithm {algorithm}")

    if algorithm in ("SAC", "DDPG"):
        parser.add_argument(
            "--annealing_frames",
            type=int,
            default=1000000,
            help="Number of frames used for annealing of the OrnsteinUhlenbeckProcess. Default=1e6.",
        )
        parser.add_argument(
            "--noisy",
            action="store_true",
            help="whether to use NoisyLinearLayers in the value network.",
        )
        parser.add_argument(
            "--ou_exploration",
            action="store_true",
            help="wraps the policy in an OU exploration wrapper, similar to DDPG. SAC being designed for "
                 "efficient entropy-based exploration, this should be left for experimentation only.",
        )

    if algorithm == "SAC":
        parser.add_argument(
            "--single_qvalue",
            action="store_false",
            dest="double_qvalue",
            help="As suggested in the original SAC paper and in https://arxiv.org/abs/1802.09477, we can "
                 "use two different qvalue networks trained independently and choose the lowest value "
                 "predicted to predict the state action value. This can be disabled by using this flag.",
        )

    if algorithm in ("SAC", "PPO"):
        parser.add_argument(
            "--tanh_normal_tanh",
            "--tanh-normal-tanh",
            action="store_true",
            help="if True, uses a Tanh-Normal transform for the policy location",
        )
        parser.add_argument(
            "--default_policy_scale",
            default=1.0,
            help="Default policy scale parameter",
        )
        parser.add_argument(
            "--distribution",
            type=str,
            default="tanh_normal",
            help="if True, uses a Tanh-Normal-Tanh distribution for the policy",
        )
    if algorithm == "PPO":
        parser.add_argument(
            "--lstm",
            action="store_true",
            help="if True, uses an LSTM for the policy.",
        )

    return parser


def parser_model_args_discrete(parser: ArgumentParser) -> ArgumentParser:
    """
    To be used for DQN, Rainbow
    """
    parser.add_argument(
        "--annealing_frames",
        type=int,
        default=1000000,
        help="Number of frames used for annealing of the EGreedy exploration. Default=1e6.",
    )

    parser.add_argument(
        "--noisy",
        action="store_true",
        help="whether to use NoisyLinearLayers in the value network.",
    )

    return parser
