from argparse import Namespace, ArgumentParser
from numbers import Number
from typing import Optional, Iterable

import torch
from torch import nn

from torchrl.data import UnboundedContinuousTensorSpec, DEVICE_TYPING
from torchrl.envs.common import _EnvClass
from torchrl.modules import ActorValueOperator, NoisyLinear, TDModule
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
    ConvNet,
    LSTMNet,
)
from torchrl.modules.td_module import (
    QValueActor,
    DistributionalQValueActor,
    Actor,
    ProbabilisticTDModule,
)
from torchrl.modules.td_module.actors import (
    ValueOperator,
    ActorCriticWrapper,
    ProbabilisticActor,
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
    "make_redq_model",
    "parser_model_args_continuous",
    "parser_model_args_discrete",
]


def make_dqn_actor(
    proof_environment: _EnvClass, device: torch.device, args: Namespace
) -> Actor:
    """
    DQN constructor helper function.

    Args:
        proof_environment (_EnvClass): a dummy environment to retrieve the observation and action spec.
        device (torch.device): device on which the model must be cast
        args (argparse.Namespace): arguments of the DQN script

    Returns: A DQN policy operator.

    Examples:
        >>> from torchrl.agents.helpers.models import make_dqn_actor, parser_model_args_discrete
        >>> from torchrl.envs import GymEnv
        >>> from torchrl.data.transforms import ToTensorImage, TransformedEnv
        >>> import argparse
        >>> proof_environment = TransformedEnv(GymEnv("Pong-v0", pixels_only=True), ToTensorImage())
        >>> device = torch.device("cpu")
        >>> args = parser_model_args_discrete(argparse.ArgumentParser()).parse_args([])
        >>> actor = make_dqn_actor(proof_environment, device, args)
        >>> td = proof_environment.reset()
        >>> print(actor(td))
        TensorDict(
            fields={done: Tensor(torch.Size([1]), dtype=torch.bool),
                observation_pixels: Tensor(torch.Size([3, 210, 160]), dtype=torch.float32),
                action: Tensor(torch.Size([6]), dtype=torch.int64),
                action_value: Tensor(torch.Size([6]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([]),
            device=cpu)


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

    model = actor_class(
        spec=env_specs["action_spec"],
        in_keys=[in_key],
        module=net,
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
    from_pixels: bool,
    noisy: bool,
    actor_net_kwargs: Optional[dict] = None,
    value_net_kwargs: Optional[dict] = None,
    atoms: int = 0,  # for distributional dqn
    vmin: Number = -3,
    vmax: Number = 3,
    device: DEVICE_TYPING = "cpu",
) -> torch.nn.ModuleList:
    """
    DDPG constructor helper function.

    Args:
        proof_environment (_EnvClass): a dummy environment to retrieve the observation and action spec
        from_pixels (bool): if True, data is assumed to be an image content.
        noisy (bool): whether or not to use noisy linear layers.
        actor_net_kwargs (dict, optional): kwargs to be used for the policy network (either DdpgCnnActor or
            DdpgMlpActor).
        value_net_kwargs (dict, optional): kwargs to be used for the policy network (either DdpgCnnQNet or
            DdpgMlpQNet).
        atoms (int, optional): not implemented.
        vmin (scalar, optional): not implemented.
        vmax (scalar, optional): not implemented.
        device (torch.device, optional): device on which the model must be cast. Default is "cpu".

    Returns: An actor and a value operators for DDPG.

    For more details on DDPG, refer to "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING",
    https://arxiv.org/pdf/1509.02971.pdf.

    Examples:
        >>> from torchrl.agents.helpers.models import make_ddpg_actor, parser_model_args_continuous
        >>> from torchrl.envs import GymEnv
        >>> from torchrl.data.transforms import CatTensors, TransformedEnv, DoubleToFloat, Compose
        >>> import argparse
        >>> proof_environment = TransformedEnv(GymEnv("HalfCheetah-v2"), Compose(DoubleToFloat(["next_observation"]),
        ...    CatTensors(["next_observation"], "next_observation_vector")))
        >>> device = torch.device("cpu")
        >>> args = parser_model_args_continuous(argparse.ArgumentParser(), algorithm="DDPG").parse_args([])
        >>> actor, value = make_ddpg_actor(
        ...     proof_environment,
        ...     device=device,
        ...     from_pixels=False,
        ...     noisy=args.noisy)
        >>> td = proof_environment.reset()
        >>> print(actor(td))
        TensorDict(
            fields={done: Tensor(torch.Size([1]), dtype=torch.bool),
                observation_vector: Tensor(torch.Size([17]), dtype=torch.float32),
                action: Tensor(torch.Size([6]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([]),
            device=cpu)
        >>> print(value(td))
        TensorDict(
            fields={done: Tensor(torch.Size([1]), dtype=torch.bool),
                observation_vector: Tensor(torch.Size([17]), dtype=torch.float32),
                action: Tensor(torch.Size([6]), dtype=torch.float32),
                state_action_value: Tensor(torch.Size([1]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([]),
            device=cpu)
    """

    actor_net_kwargs = actor_net_kwargs if actor_net_kwargs is not None else dict()
    value_net_kwargs = value_net_kwargs if value_net_kwargs is not None else dict()

    linear_layer_class = torch.nn.Linear if not noisy else NoisyLinear

    env_specs = proof_environment.specs
    out_features = env_specs["action_spec"].shape[0]

    # We use a ProbabilisticActor to make sure that we map the network output to the right space using a TanhDelta
    # distribution.
    actor_class = ProbabilisticActor
    if atoms:
        raise NotImplementedError
        # https://arxiv.org/pdf/1804.08617.pdf

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
        spec=env_specs["action_spec"],
        module=actor_net,
        safe=True,
        distribution_class=TanhDelta,
        distribution_kwargs={
            "min": env_specs["action_spec"].space.minimum,
            "max": env_specs["action_spec"].space.maximum,
        },
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
        module=q_net,
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
    args: Namespace,
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
        proof_environment (_EnvClass): a dummy environment to retrieve the observation and action spec
        args (argparse.Namespace): arguments of the PPO script
        device (torch.device): device on which the model must be cast.
        in_keys_actor (iterable of strings, optional): observation key to be read by the actor, usually one of
            `'observation_vector'` or `'observation_pixels'`. If none is provided, one of these two keys is chosen based on
            the `args.from_pixels` argument.

    Returns: A joined ActorCriticOperator.

    Examples:
        >>> from torchrl.agents.helpers.models import make_ppo_model, parser_model_args_continuous
        >>> from torchrl.envs import GymEnv
        >>> from torchrl.data.transforms import CatTensors, TransformedEnv, DoubleToFloat, Compose
        >>> import argparse
        >>> proof_environment = TransformedEnv(GymEnv("HalfCheetah-v2"), Compose(DoubleToFloat(["next_observation"]),
        ...    CatTensors(["next_observation"], "next_observation_vector")))
        >>> device = torch.device("cpu")
        >>> args = parser_model_args_continuous(
        ...         argparse.ArgumentParser(), algorithm="PPO").parse_args(["--shared_mapping"])
        >>> actor_value = make_ppo_model(
        ...     proof_environment,
        ...     device=device,
        ...     args=args,
        ...     )
        >>> actor = actor_value.get_policy_operator()
        >>> value = actor_value.get_value_operator()
        >>> td = proof_environment.reset()
        >>> print(actor(td.clone()))
        TensorDict(
            fields={done: Tensor(torch.Size([1]), dtype=torch.bool),
                observation_vector: Tensor(torch.Size([17]), dtype=torch.float32),
                hidden: Tensor(torch.Size([300]), dtype=torch.float32),
                action_dist_param_0: Tensor(torch.Size([12]), dtype=torch.float32),
                action: Tensor(torch.Size([6]), dtype=torch.float32),
                action_log_prob: Tensor(torch.Size([1]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([]),
            device=cpu)
        >>> print(value(td.clone()))
        TensorDict(
            fields={done: Tensor(torch.Size([1]), dtype=torch.bool),
                observation_vector: Tensor(torch.Size([17]), dtype=torch.float32),
                hidden: Tensor(torch.Size([300]), dtype=torch.float32),
                state_value: Tensor(torch.Size([1]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([]),
            device=cpu)

    """
    # proof_environment.set_seed(args.seed)
    specs = proof_environment.specs  # TODO: use env.sepcs
    action_spec = specs["action_spec"]

    if in_keys_actor is None and proof_environment.from_pixels:
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
                "tanh_loc": args.tanh_loc,
                "scale_mapping": f"biased_softplus_{args.default_policy_scale}",
            }
            policy_distribution_class = TanhNormal
        elif args.distribution == "truncated_normal":
            policy_distribution_kwargs = {
                "min": action_spec.space.minimum,
                "max": action_spec.space.maximum,
                "tanh_loc": args.tanh_loc,
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
        if proof_environment.from_pixels:
            if in_keys_actor is None:
                in_keys_actor = ["observation_pixels"]
            common_module = ConvNet(
                bias_last_layer=True,
                depth=None,
                num_cells=[32, 64, 64],
                kernel_sizes=[8, 4, 3],
                strides=[4, 2, 1],
            )
        else:
            if args.lstm:
                raise NotImplementedError(
                    "lstm not yet compatible with shared mapping for PPO"
                )
            common_module = MLP(
                num_cells=[
                    400,
                ],
                out_features=300,
                activate_last_layer=True,
            )
        common_operator = TDModule(
            spec=None, module=common_module, in_keys=in_keys_actor, out_keys=["hidden"]
        )

        policy_net = MLP(
            num_cells=[200],
            out_features=out_features,
        )
        policy_operator = ProbabilisticActor(
            spec=action_spec,
            module=policy_net,
            in_keys=["hidden"],
            default_interaction_mode="random",
            distribution_class=policy_distribution_class,
            distribution_kwargs=policy_distribution_kwargs,
            return_log_prob=True,
            save_dist_params=True,
        )
        value_net = MLP(
            num_cells=[200],
            out_features=1,
        )
        value_operator = ValueOperator(value_net, in_keys=["hidden"])
        actor_value = ActorValueOperator(
            common_operator=common_operator,
            policy_operator=policy_operator,
            value_operator=value_operator,
        ).to(device)
    else:
        if args.lstm:
            policy_net = LSTMNet(
                out_features=out_features,
                lstm_kwargs={"input_size": 256, "hidden_size": 256},
                mlp_kwargs={"num_cells": [256, 256], "out_features": 256},
            )
            in_keys_actor += ["hidden0", "hidden1"]
            out_keys += ["hidden0", "hidden1", "next_hidden0", "next_hidden1"]
        else:
            policy_net = MLP(
                num_cells=[400, 300],
                out_features=out_features,
            )

        policy_po = ProbabilisticActor(
            action_spec,
            policy_net,
            distribution_class=policy_distribution_class,
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
    device: DEVICE_TYPING = "cpu",
    tanh_loc: bool = True,
    default_policy_scale: float = 1.0,
    **kwargs,
) -> nn.ModuleList:
    """
    Actor, Q-value and value model constructor helper function for SAC.
    Follows default parameters proposed in SAC original paper: https://arxiv.org/pdf/1801.01290.pdf.
    Other configurations can easily be implemented by modifying this function at will.

    Args:
        proof_environment (_EnvClass): a dummy environment to retrieve the observation and action spec
        in_keys (iterable of strings, optional): observation key to be read by the actor, usually one of
            `'observation_vector'` or `'observation_pixels'`. If none is provided, one of these two keys is chosen
             based on the `args.from_pixels` argument.
        actor_net_kwargs (dict, optional): kwargs of the actor MLP.
        qvalue_net_kwargs (dict, optional): kwargs of the qvalue MLP.
        value_net_kwargs (dict, optional): kwargs of the value MLP.
        device (torch.device, optional): device on which the model must be cast. Default is "cpu".
        tanh_loc (bool, optional): whether to use a tanh scaling for the distribution location parameter.
            Default is True.
        default_policy_scale (positive scalar, optional): Default scale of the policy distribution (i.e. standard
            deviation of the normal distribution when the network output is 0). Caution: a higher standard
            deviation may not lead to a more entropic distribution, as a Tanh transform is applied to the
            generated variables. The maximum entropy configuration is with a standard deviation of 0.87. Default is 1.0.
    Returns: A nn.ModuleList containing the actor, qvalue operator(s) and the value operator.

    Examples:
        >>> from torchrl.agents.helpers.models import make_sac_model, parser_model_args_continuous
        >>> from torchrl.envs import GymEnv
        >>> from torchrl.data.transforms import CatTensors, TransformedEnv, DoubleToFloat, Compose
        >>> import argparse
        >>> proof_environment = TransformedEnv(GymEnv("HalfCheetah-v2"), Compose(DoubleToFloat(["next_observation"]),
        ...    CatTensors(["next_observation"], "next_observation_vector")))
        >>> device = torch.device("cpu")
        >>> args = parser_model_args_continuous(
        ...         argparse.ArgumentParser(), algorithm="SAC").parse_args([])
        >>> model = make_sac_model(
        ...     proof_environment,
        ...     device=device,
        ...     )
        >>> actor, qvalue, value = model
        >>> td = proof_environment.reset()
        >>> print(actor(td))
        TensorDict(
            fields={
                done: Tensor(torch.Size([1]), dtype=torch.bool),
                observation_vector: Tensor(torch.Size([17]), dtype=torch.float32),
                action: Tensor(torch.Size([6]), dtype=torch.float32)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)
        >>> print(qvalue(td.clone()))
        TensorDict(
            fields={
                done: Tensor(torch.Size([1]), dtype=torch.bool),
                observation_vector: Tensor(torch.Size([17]), dtype=torch.float32),
                action: Tensor(torch.Size([6]), dtype=torch.float32),
                state_action_value: Tensor(torch.Size([1]), dtype=torch.float32)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)
        >>> print(value(td.clone()))
        TensorDict(
            fields={
                done: Tensor(torch.Size([1]), dtype=torch.bool),
                observation_vector: Tensor(torch.Size([17]), dtype=torch.float32),
                action: Tensor(torch.Size([6]), dtype=torch.float32),
                state_value: Tensor(torch.Size([1]), dtype=torch.float32)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)

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

    value_net_kwargs_default = {
        "num_cells": [256, 256],
        "out_features": 1,
        "activation_class": nn.ELU,
    }
    value_net_kwargs_default.update(value_net_kwargs)
    value_net = MLP(
        **value_net_kwargs_default,
    )

    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=in_keys,
        module=actor_net,
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": action_spec.space.minimum,
            "max": action_spec.space.maximum,
            "tanh_loc": tanh_loc,
            "scale_mapping": f"biased_softplus_{default_policy_scale}",
        },
        default_interaction_mode="random",
    )
    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )
    value = ValueOperator(
        in_keys=in_keys,
        module=value_net,
    )
    model = nn.ModuleList([actor, qvalue, value]).to(device)

    # init nets
    td = td.to(device)
    for net in model:
        net(td)
    del td

    proof_environment.close()

    return model


def make_redq_model(
    proof_environment: _EnvClass,
    in_keys: Optional[Iterable[str]] = None,
    actor_net_kwargs=None,
    qvalue_net_kwargs=None,
    device: DEVICE_TYPING = "cpu",
    tanh_loc: bool = True,
    default_policy_scale: float = 1.0,
    **kwargs,
) -> nn.ModuleList:
    """
    Actor and Q-value model constructor helper function for REDQ.
    Follows default parameters proposed in REDQ original paper: https://openreview.net/pdf?id=AY8zfZm0tDd.
    Other configurations can easily be implemented by modifying this function at will.
    A single instance of the Q-value model is returned. It will be multiplicated by the loss function.

    Args:
        proof_environment (_EnvClass): a dummy environment to retrieve the observation and action spec
        in_keys (iterable of strings, optional): observation key to be read by the actor, usually one of
            `'observation_vector'` or `'observation_pixels'`. If none is provided, one of these two keys is chosen
             based on the `args.from_pixels` argument.
        actor_net_kwargs (dict, optional): kwargs of the actor MLP.
        qvalue_net_kwargs (dict, optional): kwargs of the qvalue MLP.
        device (torch.device, optional): device on which the model must be cast. Default is "cpu".
        tanh_loc (bool, optional): whether to use a tanh scaling for the distribution location parameter.
            Default is True.
        default_policy_scale (positive scalar, optional): Default scale of the policy distribution (i.e. standard
            deviation of the normal distribution when the network output is 0). Caution: a higher standard
            deviation may not lead to a more entropic distribution, as a Tanh transform is applied to the
            generated variables. The maximum entropy configuration is with a standard deviation of 0.87. Default is 1.0.
    Returns: A nn.ModuleList containing the actor, qvalue operator(s) and the value operator.

    Examples:
        >>> from torchrl.agents.helpers.models import make_redq_model, parser_model_args_continuous
        >>> from torchrl.envs import GymEnv
        >>> from torchrl.data.transforms import CatTensors, TransformedEnv, DoubleToFloat, Compose
        >>> import argparse
        >>> proof_environment = TransformedEnv(GymEnv("HalfCheetah-v2"), Compose(DoubleToFloat(["next_observation"]),
        ...    CatTensors(["next_observation"], "next_observation_vector")))
        >>> device = torch.device("cpu")
        >>> args = parser_model_args_continuous(
        ...         argparse.ArgumentParser(), algorithm="REDQ").parse_args([])
        >>> model = make_redq_model(
        ...     proof_environment,
        ...     device=device,
        ...     )
        >>> actor, qvalue = model
        >>> td = proof_environment.reset()
        >>> print(actor(td))
        TensorDict(
            fields={
                done: Tensor(torch.Size([1]), dtype=torch.bool),
                observation_vector: Tensor(torch.Size([17]), dtype=torch.float32),
                action: Tensor(torch.Size([6]), dtype=torch.float32)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)
        >>> print(qvalue(td.clone()))
        TensorDict(
            fields={
                done: Tensor(torch.Size([1]), dtype=torch.bool),
                observation_vector: Tensor(torch.Size([17]), dtype=torch.float32),
                action: Tensor(torch.Size([6]), dtype=torch.float32),
                state_action_value: Tensor(torch.Size([1]), dtype=torch.float32)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)

    """
    td = proof_environment.reset()
    action_spec = proof_environment.action_spec
    if actor_net_kwargs is None:
        actor_net_kwargs = {}
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

    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=in_keys,
        module=actor_net,
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": action_spec.space.minimum,
            "max": action_spec.space.maximum,
            "tanh_loc": tanh_loc,
            "scale_mapping": f"biased_softplus_{default_policy_scale}",
        },
        default_interaction_mode="random",
        return_log_prob=True,
    )
    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )
    model = nn.ModuleList([actor, qvalue]).to(device)

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

    if algorithm not in ("SAC", "DDPG", "PPO", "REDQ"):
        raise NotImplementedError(f"Unknown algorithm {algorithm}")

    if algorithm in ("SAC", "DDPG", "REDQ"):
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
        parser.add_argument(
            "--distributional",
            action="store_true",
            help="whether a distributional loss should be used (TODO: not implemented yet).",
        )
        parser.add_argument(
            "--atoms",
            type=int,
            default=51,
            help="number of atoms used for the distributional loss (TODO)",
        )

    if algorithm in ("SAC", "PPO", "REDQ"):
        parser.add_argument(
            "--tanh_loc",
            "--tanh-loc",
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
        parser.add_argument(
            "--shared_mapping",
            "--shared-mapping",
            action="store_true",
            help="if True, the first layers of the actor-critic are shared.",
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
    parser.add_argument(
        "--distributional",
        action="store_true",
        help="whether a distributional loss should be used.",
    )
    parser.add_argument(
        "--atoms",
        type=int,
        default=51,
        help="number of atoms used for the distributional loss",
    )

    return parser
