# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import contextlib
import functools
import importlib.util
import itertools
import operator
import sys
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
import pytest
import torch

from packaging import version, version as pack_version
from tensordict import assert_allclose_td, TensorDict, TensorDictBase
from tensordict._C import unravel_keys
from tensordict.nn import (
    composite_lp_aggregate,
    CompositeDistribution,
    InteractionType,
    NormalParamExtractor,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictModule as ProbMod,
    ProbabilisticTensorDictSequential,
    ProbabilisticTensorDictSequential as ProbSeq,
    set_composite_lp_aggregate,
    TensorDictModule,
    TensorDictModule as Mod,
    TensorDictSequential,
    TensorDictSequential as Seq,
    WrapModule,
)
from tensordict.nn.distributions.composite import _add_suffix
from tensordict.nn.utils import Buffer
from tensordict.utils import unravel_key
from torch import autograd, nn

from torchrl._utils import _standardize, rl_warnings
from torchrl.data import (
    Bounded,
    Categorical,
    Composite,
    LazyTensorStorage,
    MultiOneHot,
    OneHot,
    TensorDictPrioritizedReplayBuffer,
    Unbounded,
)
from torchrl.data.postprocs.postprocs import MultiStep
from torchrl.envs import EnvBase, GymEnv, InitTracker, SerialEnv
from torchrl.envs.libs.gym import _has_gym
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.transforms import TensorDictPrimer, TransformedEnv
from torchrl.envs.utils import exploration_type, ExplorationType, set_exploration_type
from torchrl.modules import (
    DistributionalQValueActor,
    GRUModule,
    LSTMModule,
    OneHotCategorical,
    QValueActor,
    recurrent_mode,
    SafeSequential,
    set_recurrent_mode,
    WorldModelWrapper,
)
from torchrl.modules.distributions.continuous import TanhDelta, TanhNormal
from torchrl.modules.models import QMixer
from torchrl.modules.models.model_based import (
    DreamerActor,
    ObsDecoder,
    ObsEncoder,
    RSSMPosterior,
    RSSMPrior,
    RSSMRollout,
)
from torchrl.modules.models.models import MLP
from torchrl.modules.tensordict_module.actors import (
    Actor,
    ActorCriticOperator,
    ActorValueOperator,
    ProbabilisticActor,
    QValueModule,
    ValueOperator,
)
from torchrl.objectives import (
    A2CLoss,
    ClipPPOLoss,
    CQLLoss,
    CrossQLoss,
    DDPGLoss,
    DiscreteCQLLoss,
    DiscreteIQLLoss,
    DiscreteSACLoss,
    DistributionalDQNLoss,
    DQNLoss,
    DreamerActorLoss,
    DreamerModelLoss,
    DreamerValueLoss,
    DTLoss,
    GAILLoss,
    IQLLoss,
    KLPENPPOLoss,
    OnlineDTLoss,
    PPOLoss,
    QMixerLoss,
    SACLoss,
    TD3BCLoss,
    TD3Loss,
)
from torchrl.objectives.common import add_random_module, LossModule
from torchrl.objectives.deprecated import DoubleREDQLoss_deprecated, REDQLoss_deprecated
from torchrl.objectives.redq import REDQLoss
from torchrl.objectives.reinforce import ReinforceLoss
from torchrl.objectives.utils import (
    _sum_td_features,
    _vmap_func,
    HardUpdate,
    hold_out_net,
    SoftUpdate,
    ValueEstimators,
)
from torchrl.objectives.value.advantages import (
    GAE,
    TD0Estimator,
    TD1Estimator,
    TDLambdaEstimator,
    VTrace,
)
from torchrl.objectives.value.functional import (
    _transpose_time,
    generalized_advantage_estimate,
    reward2go,
    td0_advantage_estimate,
    td1_advantage_estimate,
    td_lambda_advantage_estimate,
    vec_generalized_advantage_estimate,
    vec_td1_advantage_estimate,
    vec_td_lambda_advantage_estimate,
    vtrace_advantage_estimate,
)
from torchrl.objectives.value.utils import (
    _custom_conv1d,
    _get_num_per_traj,
    _get_num_per_traj_init,
    _inv_pad_sequence,
    _make_gammas_tensor,
    _split_and_pad_sequence,
)

from torchrl.testing import (  # noqa
    call_value_nets as _call_value_nets,
    dtype_fixture,
    get_available_devices,
    get_default_devices,
    PENDULUM_VERSIONED,
)
from torchrl.testing.mocking_classes import ContinuousActionConvMockEnv

_has_functorch = True
try:
    import functorch as ft  # noqa

    make_functional_with_buffers = ft.make_functional_with_buffers
    FUNCTORCH_ERR = ""
except ImportError as err:
    _has_functorch = False
    FUNCTORCH_ERR = str(err)

_has_transformers = bool(importlib.util.find_spec("transformers"))

TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)
IS_WINDOWS = sys.platform == "win32"

# Capture all warnings
pytestmark = [
    pytest.mark.filterwarnings("error"),
    pytest.mark.filterwarnings(
        "ignore:The current behavior of MLP when not providing `num_cells` is that the number"
    ),
    pytest.mark.filterwarnings(
        "ignore:dep_util is Deprecated. Use functions from setuptools instead"
    ),
    pytest.mark.filterwarnings(
        "ignore:The PyTorch API of nested tensors is in prototype"
    ),
    pytest.mark.filterwarnings("ignore:unclosed event loop:ResourceWarning"),
    pytest.mark.filterwarnings("ignore:unclosed.*socket:ResourceWarning"),
]


class _check_td_steady:
    def __init__(self, td):
        self.td_clone = td.clone()
        self.td = td

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert_allclose_td(
            self.td,
            self.td_clone,
            intersection=True,
            msg="Some keys have been modified in the tensordict!",
        )


def get_devices():
    devices = [torch.device("cpu")]
    for i in range(torch.cuda.device_count()):
        devices += [torch.device(f"cuda:{i}")]
    return devices


class MARLEnv(EnvBase):
    def __init__(self):
        batch = self.batch = (3,)
        super().__init__(batch_size=batch)
        self.n_agents = n_agents = (4,)
        self.obs_feat = obs_feat = (5,)

        self.full_observation_spec = Composite(
            agents=Composite(
                observation=Unbounded(batch + n_agents + obs_feat),
                shape=batch + n_agents,
            ),
            shape=batch,
        )
        self.full_done_spec = Composite(
            done=Unbounded(batch + (1,), dtype=torch.bool),
            terminated=Unbounded(batch + (1,), dtype=torch.bool),
            truncated=Unbounded(batch + (1,), dtype=torch.bool),
            shape=batch,
        )

        self.act_feat_dirich = act_feat_dirich = (10, 2)
        self.act_feat_categ = act_feat_categ = (7,)
        self.full_action_spec = Composite(
            agents=Composite(
                dirich=Unbounded(batch + n_agents + act_feat_dirich),
                categ=Unbounded(batch + n_agents + act_feat_categ),
                shape=batch + n_agents,
            ),
            shape=batch,
        )

        self.full_reward_spec = Composite(
            agents=Composite(
                reward=Unbounded(batch + n_agents + (1,)), shape=batch + n_agents
            ),
            shape=batch,
        )

    @classmethod
    def make_composite_dist(cls):
        dist_cstr = functools.partial(
            CompositeDistribution,
            distribution_map={
                (
                    "agents",
                    "dirich",
                ): lambda concentration: torch.distributions.Independent(
                    torch.distributions.Dirichlet(concentration), 1
                ),
                ("agents", "categ"): torch.distributions.Categorical,
            },
        )
        return ProbabilisticTensorDictModule(
            in_keys=["params"],
            out_keys=[("agents", "dirich"), ("agents", "categ")],
            distribution_class=dist_cstr,
            return_log_prob=True,
        )

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        ...

    def _reset(self, tensordic):
        ...

    def _set_seed(self, seed: int | None) -> None:
        ...


class LossModuleTestBase:
    @pytest.fixture(scope="class", autouse=True)
    def _composite_log_prob(self):
        setter = set_composite_lp_aggregate(False)
        setter.set()
        yield
        setter.unset()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        assert hasattr(
            cls, "test_reset_parameters_recursive"
        ), "Please add a test_reset_parameters_recursive test for this class"

    def _flatten_in_keys(self, in_keys):
        return [
            in_key if isinstance(in_key, str) else "_".join(list(unravel_keys(in_key)))
            for in_key in in_keys
        ]

    def tensordict_keys_test(self, loss_fn, default_keys, td_est=None):
        self.tensordict_keys_unknown_key_test(loss_fn)
        self.tensordict_keys_default_values_test(loss_fn, default_keys)
        self.tensordict_set_keys_test(loss_fn, default_keys)

    def tensordict_keys_unknown_key_test(self, loss_fn):
        """Test that exception is raised if an unknown key is set via .set_keys()"""
        test_fn = deepcopy(loss_fn)

        with pytest.raises(ValueError):
            test_fn.set_keys(unknown_key="test2")

    def tensordict_keys_default_values_test(self, loss_fn, default_keys):
        test_fn = deepcopy(loss_fn)

        for key, value in default_keys.items():
            assert getattr(test_fn.tensor_keys, key) == value

    def tensordict_set_keys_test(self, loss_fn, default_keys):
        """Test setting of tensordict keys via .set_keys()"""
        test_fn = deepcopy(loss_fn)

        new_key = "test1"
        for key, _ in default_keys.items():
            test_fn.set_keys(**{key: new_key})
            assert getattr(test_fn.tensor_keys, key) == new_key

        test_fn = deepcopy(loss_fn)
        test_fn.set_keys(**{key: new_key for key, _ in default_keys.items()})

        for key, _ in default_keys.items():
            assert getattr(test_fn.tensor_keys, key) == new_key

    def set_advantage_keys_through_loss_test(
        self, loss_fn, td_est, loss_advantage_key_mapping
    ):
        key_mapping = loss_advantage_key_mapping
        test_fn = deepcopy(loss_fn)

        new_keys = {}
        for loss_key, (_, new_key) in key_mapping.items():
            new_keys[loss_key] = new_key

        test_fn.set_keys(**new_keys)
        test_fn.make_value_estimator(td_est)

        for _, (advantage_key, new_key) in key_mapping.items():
            assert (
                getattr(test_fn.value_estimator.tensor_keys, advantage_key) == new_key
            )

    @classmethod
    def reset_parameters_recursive_test(cls, loss_fn):
        def get_params(loss_fn):
            for key, item in loss_fn.__dict__.items():
                if isinstance(item, nn.Module):
                    module_name = key
                    params_name = f"{module_name}_params"
                    target_name = f"target_{module_name}_params"
                    params = loss_fn._modules.get(params_name, None)
                    target = loss_fn._modules.get(target_name, None)

                    if params is not None:
                        yield params_name, params._param_td

                    else:
                        for subparam_name, subparam in loss_fn.named_parameters():
                            if module_name in subparam_name:
                                yield subparam_name, subparam

                    if target is not None:
                        yield target_name, target

        old_params = {}

        for param_name, param in get_params(loss_fn):
            with torch.no_grad():
                # Change the parameter to ensure that reset will change it again
                param += 1000
            old_params[param_name] = param.clone()

        loss_fn.reset_parameters_recursive()

        for param_name, param in get_params(loss_fn):
            old_param = old_params[param_name]
            assert (param != old_param).any()


@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("vmap_randomness", (None, "different", "same", "error"))
@pytest.mark.parametrize("dropout", (0.0, 0.1))
def test_loss_vmap_random(device, vmap_randomness, dropout):
    class VmapTestLoss(LossModule):
        model: TensorDictModule
        model_params: TensorDict
        target_model_params: TensorDict

        def __init__(self):
            super().__init__()
            layers = [nn.Linear(4, 4), nn.ReLU()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(4, 4))
            net = nn.Sequential(*layers).to(device)
            model = TensorDictModule(net, in_keys=["obs"], out_keys=["action"])
            self.convert_to_functional(model, "model", expand_dim=4)
            self._make_vmap()

        def _make_vmap(self):
            self.vmap_model = _vmap_func(
                self.model,
                (None, 0),
                randomness=(
                    "error" if vmap_randomness == "error" else self.vmap_randomness
                ),
            )

        def forward(self, td):
            out = self.vmap_model(td, self.model_params)
            return {"loss": out["action"].mean()}

    loss_module = VmapTestLoss()
    td = TensorDict({"obs": torch.randn(3, 4).to(device)}, [3])

    # If user sets vmap randomness to a specific value
    if vmap_randomness in ("different", "same") and dropout > 0.0:
        loss_module.set_vmap_randomness(vmap_randomness)
    # Fail case
    elif vmap_randomness == "error" and dropout > 0.0:
        with pytest.raises(
            RuntimeError,
            match="vmap: called random operation while in randomness error mode",
        ):
            loss_module(td)["loss"]
        return
    loss_module(td)["loss"]


class TestDQN(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(
        self,
        action_spec_type,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        is_nn_module=False,
        action_value_key=None,
    ):
        # Actor
        if action_spec_type == "one_hot":
            action_spec = OneHot(action_dim)
        elif action_spec_type == "categorical":
            action_spec = Categorical(action_dim)
        # elif action_spec_type == "nd_bounded":
        #     action_spec = Bounded(
        #         -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        #     )
        else:
            raise ValueError(f"Wrong {action_spec_type}")

        module = nn.Linear(obs_dim, action_dim)
        if is_nn_module:
            return module.to(device)
        actor = QValueActor(
            spec=Composite(
                {
                    "action": action_spec,
                    (
                        "action_value" if action_value_key is None else action_value_key
                    ): None,
                    "chosen_action_value": None,
                },
                shape=[],
            ),
            action_space=action_spec_type,
            module=module,
            action_value_key=action_value_key,
        ).to(device)
        return actor

    def _create_mock_distributional_actor(
        self,
        action_spec_type,
        batch=2,
        obs_dim=3,
        action_dim=4,
        atoms=5,
        vmin=1,
        vmax=5,
        is_nn_module=False,
        action_value_key="action_value",
    ):
        # Actor
        var_nums = None
        if action_spec_type == "mult_one_hot":
            action_spec = MultiOneHot([action_dim // 2, action_dim // 2])
            var_nums = action_spec.nvec
        elif action_spec_type == "one_hot":
            action_spec = OneHot(action_dim)
        elif action_spec_type == "categorical":
            action_spec = Categorical(action_dim)
        else:
            raise ValueError(f"Wrong {action_spec_type}")
        support = torch.linspace(vmin, vmax, atoms, dtype=torch.float)
        module = MLP(obs_dim, (atoms, action_dim))
        # TODO: Fails tests with
        # TypeError: __init__() missing 1 required keyword-only argument: 'support'
        # DistributionalQValueActor initializer expects additional inputs.
        # if is_nn_module:
        #     return module
        actor = DistributionalQValueActor(
            spec=Composite(
                {
                    "action": action_spec,
                    action_value_key: None,
                },
                shape=[],
            ),
            module=module,
            support=support,
            action_space=action_spec_type,
            var_nums=var_nums,
            action_value_key=action_value_key,
        )
        return actor

    def _create_mock_data_dqn(
        self,
        action_spec_type,
        batch=2,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
        action_key="action",
        action_value_key="action_value",
    ):
        # create a tensordict
        obs = torch.randn(batch, obs_dim)
        next_obs = torch.randn(batch, obs_dim)
        if atoms:
            action_value = torch.randn(batch, atoms, action_dim).softmax(-2)
            action = (
                action_value[..., 0, :] == action_value[..., 0, :].max(-1, True)[0]
            ).to(torch.long)
        else:
            action_value = torch.randn(batch, action_dim)
            action = (action_value == action_value.max(-1, True)[0]).to(torch.long)

        if action_spec_type == "categorical":
            action_value = torch.max(action_value, -1, keepdim=True)[0]
            action = torch.argmax(action, -1, keepdim=False)
        reward = torch.randn(batch, 1)
        done = torch.zeros(batch, 1, dtype=torch.bool)
        terminated = torch.zeros(batch, 1, dtype=torch.bool)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next": {
                    "observation": next_obs,
                    "done": done,
                    "terminated": terminated,
                    "reward": reward,
                },
                action_key: action,
                action_value_key: action_value,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_dqn(
        self,
        action_spec_type,
        batch=2,
        T=4,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim, device=device)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action_value = torch.randn(
                batch, T, atoms, action_dim, device=device
            ).softmax(-2)
            action = (
                action_value[..., 0, :] == action_value[..., 0, :].max(-1, True)[0]
            ).to(torch.long)
        else:
            action_value = torch.randn(batch, T, action_dim, device=device)
            action = (action_value == action_value.max(-1, True)[0]).to(torch.long)

        # action_value = action_value.unsqueeze(-1)
        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = ~torch.zeros(batch, T, dtype=torch.bool, device=device)
        if action_spec_type == "categorical":
            action_value = torch.max(action_value, -1, keepdim=True)[0]
            action = torch.argmax(action, -1, keepdim=False)
            action = action.masked_fill_(~mask, 0.0)
        else:
            action = action.masked_fill_(~mask.unsqueeze(-1), 0.0)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "next": {
                    "observation": next_obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                    "done": done,
                    "terminated": terminated,
                    "reward": reward.masked_fill_(~mask.unsqueeze(-1), 0.0),
                },
                "collector": {"mask": mask},
                "action": action,
                "action_value": action_value.masked_fill_(~mask.unsqueeze(-1), 0.0),
            },
            names=[None, "time"],
        )
        return td

    def test_reset_parameters_recursive(self):
        actor = self._create_mock_actor(action_spec_type="one_hot")
        loss_fn = DQNLoss(actor)
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize(
        "delay_value,double_dqn", ([False, False], [True, False], [True, True])
    )
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("action_spec_type", ("one_hot", "categorical"))
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_dqn(self, delay_value, double_dqn, device, action_spec_type, td_est):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, device=device
        )
        td = self._create_mock_data_dqn(
            action_spec_type=action_spec_type, device=device
        )
        loss_fn = DQNLoss(
            actor,
            loss_function="l2",
            delay_value=delay_value,
            double_dqn=double_dqn,
        )
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)
        with (
            pytest.warns(UserWarning, match="No target network updater has been")
            if delay_value and rl_warnings()
            else contextlib.nullcontext()
        ), _check_td_steady(td):
            loss = loss_fn(td)

        if delay_value:
            # remove warning
            SoftUpdate(loss_fn, eps=0.5)

        assert loss_fn.tensor_keys.priority in td.keys()

        sum([item for name, item in loss.items() if name.startswith("loss")]).backward()
        assert torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 1.0) > 0.0

        # Check param update effect on targets
        target_value = loss_fn.target_value_network_params.clone()
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        target_value2 = loss_fn.target_value_network_params.clone()
        if loss_fn.delay_value:
            assert_allclose_td(target_value, target_value2)
        else:
            assert not (target_value == target_value2).any()

        # check that policy is updated after parameter update
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("action_spec_type", ("one_hot", "categorical"))
    def test_dqn_state_dict(self, delay_value, device, action_spec_type):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, device=device
        )
        loss_fn = DQNLoss(actor, loss_function="l2", delay_value=delay_value)
        sd = loss_fn.state_dict()
        loss_fn2 = DQNLoss(actor, loss_function="l2", delay_value=delay_value)
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("action_spec_type", ("one_hot", "categorical"))
    def test_dqn_batcher(self, n, delay_value, device, action_spec_type, gamma=0.9):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, device=device
        )

        td = self._create_seq_mock_data_dqn(
            action_spec_type=action_spec_type, device=device
        )
        loss_fn = DQNLoss(actor, loss_function="l2", delay_value=delay_value)

        ms = MultiStep(gamma=gamma, n_steps=n).to(device)
        ms_td = ms(td.clone())

        with (
            pytest.warns(UserWarning, match="No target network updater has been")
            if delay_value and rl_warnings()
            else contextlib.nullcontext()
        ), _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)
        assert loss_fn.tensor_keys.priority in ms_td.keys()

        if delay_value:
            # remove warning
            SoftUpdate(loss_fn, eps=0.5)

        with torch.no_grad():
            loss = loss_fn(td)
        if n == 1:
            assert_allclose_td(td, ms_td.select(*td.keys(True, True)))
            _loss = sum(
                [item for name, item in loss.items() if name.startswith("loss")]
            )
            _loss_ms = sum(
                [item for name, item in loss_ms.items() if name.startswith("loss")]
            )
            assert (
                abs(_loss - _loss_ms) < 1e-3
            ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                assert_allclose_td(loss, loss_ms)
        sum(
            [item for name, item in loss_ms.items() if name.startswith("loss")]
        ).backward()
        assert torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 1.0) > 0.0

        # Check param update effect on targets
        target_value = loss_fn.target_value_network_params.clone()
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        target_value2 = loss_fn.target_value_network_params.clone()
        if loss_fn.delay_value:
            assert_allclose_td(target_value, target_value2)
        else:
            assert not (target_value == target_value2).any()

        # check that policy is updated after parameter update
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_dqn_tensordict_keys(self, td_est):
        torch.manual_seed(self.seed)
        action_spec_type = "one_hot"
        actor = self._create_mock_actor(action_spec_type=action_spec_type)
        loss_fn = DQNLoss(actor, delay_value=True)

        default_keys = {
            "advantage": "advantage",
            "value_target": "value_target",
            "value": "chosen_action_value",
            "priority": "td_error",
            "action_value": "action_value",
            "action": "action",
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
        }

        self.tensordict_keys_test(loss_fn, default_keys=default_keys)

        loss_fn = DQNLoss(actor, delay_value=True)
        key_mapping = {
            "advantage": ("advantage", "advantage_2"),
            "value_target": ("value_target", ("value_target", "nested")),
            "reward": ("reward", "reward_test"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, action_value_key="chosen_action_value_2"
        )
        loss_fn = DQNLoss(actor, delay_value=True)
        key_mapping = {
            "value": ("value", "chosen_action_value_2"),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize("action_spec_type", ("categorical", "one_hot"))
    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_dqn_tensordict_run(self, action_spec_type, td_est):
        torch.manual_seed(self.seed)
        tensor_keys = {
            "action_value": "action_value_test",
            "action": "action_test",
            "priority": "priority_test",
        }
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type,
            action_value_key=tensor_keys["action_value"],
        )
        td = self._create_mock_data_dqn(
            action_spec_type=action_spec_type,
            action_key=tensor_keys["action"],
            action_value_key=tensor_keys["action_value"],
        )

        loss_fn = DQNLoss(actor, loss_function="l2", delay_value=True)
        loss_fn.set_keys(**tensor_keys)

        if td_est is not None:
            loss_fn.make_value_estimator(td_est)

        SoftUpdate(loss_fn, eps=0.5)

        with _check_td_steady(td):
            _ = loss_fn(td)
        assert loss_fn.tensor_keys.priority in td.keys()

    @pytest.mark.parametrize("atoms", range(4, 10))
    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_devices())
    @pytest.mark.parametrize(
        "action_spec_type", ("mult_one_hot", "one_hot", "categorical")
    )
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_distributional_dqn(
        self, atoms, delay_value, device, action_spec_type, td_est, gamma=0.9
    ):
        torch.manual_seed(self.seed)
        actor = self._create_mock_distributional_actor(
            action_spec_type=action_spec_type, atoms=atoms
        ).to(device)

        td = self._create_mock_data_dqn(
            action_spec_type=action_spec_type, atoms=atoms
        ).to(device)
        loss_fn = DistributionalDQNLoss(
            actor,
            gamma=gamma,
            delay_value=delay_value,
        )

        if td_est not in (None, ValueEstimators.TD0):
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        elif td_est is not None:
            loss_fn.make_value_estimator(td_est)

        with _check_td_steady(td), (
            pytest.warns(
                UserWarning,
                match="No target network updater has been associated with this loss module",
            )
            if delay_value and rl_warnings()
            else contextlib.nullcontext()
        ):
            loss = loss_fn(td)

        assert loss_fn.tensor_keys.priority in td.keys()

        sum([item for name, item in loss.items() if name.startswith("loss")]).backward()
        assert torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 1.0) > 0.0

        if delay_value:
            # remove warning
            SoftUpdate(loss_fn, eps=0.5)

        # Check param update effect on targets
        target_value = loss_fn.target_value_network_params.clone()
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        target_value2 = loss_fn.target_value_network_params.clone()
        if loss_fn.delay_value:
            assert_allclose_td(target_value, target_value2)
        else:
            for key, val in target_value.flatten_keys(",").items():
                if "support" in key:
                    continue
                assert not (val == target_value2[tuple(key.split(","))]).any(), key

        # check that policy is updated after parameter update
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    def test_dqn_notensordict(
        self, observation_key, reward_key, done_key, terminated_key
    ):
        n_obs = 3
        n_action = 4
        action_spec = OneHot(n_action)
        module = nn.Linear(n_obs, n_action)  # a simple value model
        actor = QValueActor(
            spec=action_spec,
            action_space="one_hot",
            module=module,
            in_keys=[observation_key],
        )
        dqn_loss = DQNLoss(actor, delay_value=True)
        dqn_loss.set_keys(reward=reward_key, done=done_key, terminated=terminated_key)
        # define data
        observation = torch.randn(n_obs)
        next_observation = torch.randn(n_obs)
        action = action_spec.rand()
        next_reward = torch.randn(1)
        next_done = torch.zeros(1, dtype=torch.bool)
        next_terminated = torch.zeros(1, dtype=torch.bool)
        kwargs = {
            observation_key: observation,
            f"next_{observation_key}": next_observation,
            f"next_{reward_key}": next_reward,
            f"next_{done_key}": next_done,
            f"next_{terminated_key}": next_terminated,
            "action": action,
        }
        td = TensorDict(kwargs, []).unflatten_keys("_")
        # Disable warning
        SoftUpdate(dqn_loss, eps=0.5)
        loss_val = dqn_loss(**kwargs)
        loss_val_td = dqn_loss(td)
        torch.testing.assert_close(loss_val_td.get("loss"), loss_val)

    def test_distributional_dqn_tensordict_keys(self):
        torch.manual_seed(self.seed)
        action_spec_type = "one_hot"
        atoms = 2
        gamma = 0.9
        actor = self._create_mock_distributional_actor(
            action_spec_type=action_spec_type, atoms=atoms
        )

        loss_fn = DistributionalDQNLoss(actor, gamma=gamma, delay_value=True)

        default_keys = {
            "priority": "td_error",
            "action_value": "action_value",
            "action": "action",
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
            "steps_to_next_obs": "steps_to_next_obs",
        }

        self.tensordict_keys_test(loss_fn, default_keys=default_keys)

    @pytest.mark.parametrize("action_spec_type", ("categorical", "one_hot"))
    @pytest.mark.parametrize("td_est", [ValueEstimators.TD0])
    def test_distributional_dqn_tensordict_run(self, action_spec_type, td_est):
        torch.manual_seed(self.seed)
        atoms = 4
        tensor_keys = {
            "action_value": "action_value_test",
            "action": "action_test",
            "priority": "priority_test",
        }
        actor = self._create_mock_distributional_actor(
            action_spec_type=action_spec_type,
            atoms=atoms,
            action_value_key=tensor_keys["action_value"],
        )
        td = self._create_mock_data_dqn(
            action_spec_type=action_spec_type,
            atoms=atoms,
            action_key=tensor_keys["action"],
            action_value_key=tensor_keys["action_value"],
        )
        loss_fn = DistributionalDQNLoss(actor, gamma=0.9, delay_value=True)
        loss_fn.set_keys(**tensor_keys)

        loss_fn.make_value_estimator(td_est)

        # remove warnings
        SoftUpdate(loss_fn, eps=0.5)

        with _check_td_steady(td):
            _ = loss_fn(td)
        assert loss_fn.tensor_keys.priority in td.keys()

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_dqn_reduction(self, reduction):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        actor = self._create_mock_actor(action_spec_type="categorical", device=device)
        td = self._create_mock_data_dqn(action_spec_type="categorical", device=device)
        loss_fn = DQNLoss(
            actor,
            loss_function="l2",
            delay_value=False,
            reduction=reduction,
        )
        loss_fn.make_value_estimator()
        loss = loss_fn(td)
        if reduction == "none":
            for key in loss.keys():
                if key.startswith("loss"):
                    assert loss[key].shape == td.shape
        else:
            for key in loss.keys():
                if not key.startswith("loss"):
                    continue
                assert loss[key].shape == torch.Size([])

    @pytest.mark.parametrize("atoms", range(4, 10))
    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_distributional_dqn_reduction(self, reduction, atoms):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        actor = self._create_mock_distributional_actor(
            action_spec_type="categorical", atoms=atoms
        ).to(device)
        td = self._create_mock_data_dqn(action_spec_type="categorical", device=device)
        loss_fn = DistributionalDQNLoss(
            actor, gamma=0.9, delay_value=False, reduction=reduction
        )
        loss_fn.make_value_estimator()
        loss = loss_fn(td)
        if reduction == "none":
            for key in loss.keys():
                if key.startswith("loss"):
                    assert loss[key].shape == td.shape
        else:
            for key in loss.keys():
                if not key.startswith("loss"):
                    continue
                assert loss[key].shape == torch.Size([])

    def test_dqn_prioritized_weights(self):
        """Test DQN with prioritized replay buffer weighted loss reduction."""
        n_obs = 4
        n_actions = 3
        batch_size = 32
        buffer_size = 100

        # Create DQN value network using QValueActor
        module = nn.Linear(n_obs, n_actions)
        action_spec = Categorical(n_actions)
        value = QValueActor(
            spec=Composite(
                {
                    "action": action_spec,
                    "action_value": None,
                    "chosen_action_value": None,
                },
                shape=[],
            ),
            action_space="categorical",
            module=module,
        )

        # Create DQN loss
        loss_fn = DQNLoss(
            value_network=value, action_space="categorical", reduction="mean"
        )
        loss_fn.make_value_estimator()
        softupdate = SoftUpdate(loss_fn, eps=0.5)

        # Create prioritized replay buffer
        rb = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.9,
            storage=LazyTensorStorage(buffer_size),
            batch_size=batch_size,
            priority_key="td_error",
        )

        # Create initial data
        initial_data = TensorDict(
            {
                "observation": torch.randn(buffer_size, n_obs),
                "action": torch.randint(0, n_actions, (buffer_size,)),
                ("next", "observation"): torch.randn(buffer_size, n_obs),
                ("next", "reward"): torch.randn(buffer_size, 1),
                ("next", "done"): torch.zeros(buffer_size, 1, dtype=torch.bool),
                ("next", "terminated"): torch.zeros(buffer_size, 1, dtype=torch.bool),
            },
            batch_size=[buffer_size],
        )
        rb.extend(initial_data)

        # Sample (weights should all be identical initially)
        sample1 = rb.sample()
        assert "priority_weight" in sample1.keys()
        weights1 = sample1["priority_weight"]
        assert torch.allclose(weights1, weights1[0], atol=1e-5)

        # Run loss to get priorities
        loss_fn(sample1)
        assert "td_error" in sample1.keys()

        # Update replay buffer with new priorities
        rb.update_tensordict_priority(sample1)

        # Sample again - weights should now be non-equal
        sample2 = rb.sample()
        weights2 = sample2["priority_weight"]
        assert weights2.std() > 1e-5

        # Run loss again with varied weights
        loss_out2 = loss_fn(sample2)
        assert torch.isfinite(loss_out2["loss"])

        # Verify manual weighted average matches
        loss_fn_no_reduction = DQNLoss(
            value_network=value,
            action_space="categorical",
            reduction="none",
            use_prioritized_weights=False,
        )
        softupdate = SoftUpdate(loss_fn_no_reduction, eps=0.5)
        loss_fn_no_reduction.make_value_estimator()
        loss_fn_no_reduction.target_value_network_params = (
            loss_fn.target_value_network_params
        )

        loss_elements = loss_fn_no_reduction(sample2)["loss"]
        manual_weighted_loss = (loss_elements * weights2).sum() / weights2.sum()
        assert torch.allclose(loss_out2["loss"], manual_weighted_loss, rtol=1e-4)


class TestQMixer(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(
        self,
        action_spec_type,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        observation_key=("agents", "observation"),
        action_key=("agents", "action"),
        action_value_key=("agents", "action_value"),
        chosen_action_value_key=("agents", "chosen_action_value"),
    ):
        # Actor
        if action_spec_type == "one_hot":
            action_spec = OneHot(action_dim)
        elif action_spec_type == "categorical":
            action_spec = Categorical(action_dim)
        else:
            raise ValueError(f"Wrong {action_spec_type}")

        module = nn.Linear(obs_dim, action_dim).to(device)

        module = TensorDictModule(
            module,
            in_keys=[observation_key],
            out_keys=[action_value_key],
        ).to(device)
        value_module = QValueModule(
            action_value_key=action_value_key,
            out_keys=[
                action_key,
                action_value_key,
                chosen_action_value_key,
            ],
            spec=action_spec,
            action_space=None,
        ).to(device)
        actor = SafeSequential(module, value_module)

        return actor

    def _create_mock_mixer(
        self,
        state_shape=(64, 64, 3),
        n_agents=4,
        device="cpu",
        chosen_action_value_key=("agents", "chosen_action_value"),
        state_key="state",
        global_chosen_action_value_key="chosen_action_value",
    ):
        qmixer = TensorDictModule(
            module=QMixer(
                state_shape=state_shape,
                mixing_embed_dim=32,
                n_agents=n_agents,
                device=device,
            ),
            in_keys=[chosen_action_value_key, state_key],
            out_keys=[global_chosen_action_value_key],
        ).to(device)

        return qmixer

    def _create_mock_data_dqn(
        self,
        action_spec_type,
        batch=(2,),
        T=None,
        n_agents=4,
        obs_dim=3,
        state_shape=(64, 64, 3),
        action_dim=4,
        device="cpu",
        action_key=("agents", "action"),
        action_value_key=("agents", "action_value"),
    ):
        if T is not None:
            batch = batch + (T,)
        # create a tensordict
        obs = torch.randn(*batch, n_agents, obs_dim, device=device)
        state = torch.randn(*batch, *state_shape, device=device)
        next_obs = torch.randn(*batch, n_agents, obs_dim, device=device)
        next_state = torch.randn(*batch, *state_shape, device=device)

        action_value = torch.randn(*batch, n_agents, action_dim, device=device)
        if action_spec_type == "one_hot":
            action = (action_value == action_value.max(dim=-1, keepdim=True)[0]).to(
                torch.long
            )
        elif action_spec_type == "categorical":
            action = torch.argmax(action_value, dim=-1).to(torch.long)

        reward = torch.randn(*batch, 1, device=device)
        done = torch.zeros(*batch, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(*batch, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            {
                "agents": TensorDict(
                    {"observation": obs},
                    [*batch, n_agents],
                    device=device,
                ),
                "state": state,
                "collector": {
                    "mask": torch.zeros(*batch, dtype=torch.bool, device=device)
                },
                "next": TensorDict(
                    {
                        "agents": TensorDict(
                            {"observation": next_obs},
                            [*batch, n_agents],
                            device=device,
                        ),
                        "state": next_state,
                        "reward": reward,
                        "done": done,
                        "terminated": terminated,
                    },
                    batch_size=batch,
                    device=device,
                ),
            },
            batch_size=batch,
            device=device,
        )
        td.set(action_key, action)
        td.set(action_value_key, action_value)
        if T is not None:
            td.refine_names(None, "time")
        return td

    def test_reset_parameters_recursive(self):
        actor = self._create_mock_actor(action_spec_type="one_hot")
        mixer = self._create_mock_mixer()
        loss_fn = QMixerLoss(actor, mixer)
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("action_spec_type", ("one_hot", "categorical"))
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_qmixer(self, delay_value, device, action_spec_type, td_est):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, device=device
        )
        mixer = self._create_mock_mixer(device=device)
        td = self._create_mock_data_dqn(
            action_spec_type=action_spec_type, device=device
        )
        loss_fn = QMixerLoss(actor, mixer, loss_function="l2", delay_value=delay_value)
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)
        with (
            pytest.warns(UserWarning, match="No target network updater has been")
            if delay_value and rl_warnings()
            else contextlib.nullcontext()
        ), _check_td_steady(td):
            loss = loss_fn(td)
        assert loss_fn.tensor_keys.priority in td.keys()

        sum([item for name, item in loss.items() if name.startswith("loss")]).backward()
        assert torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 1.0) > 0.0

        if delay_value:
            # remove warning
            SoftUpdate(loss_fn, eps=0.5)

        # Check param update effect on targets
        target_value = loss_fn.target_local_value_network_params.clone()
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += 3
        target_value2 = loss_fn.target_local_value_network_params.clone()
        if loss_fn.delay_value:
            assert_allclose_td(target_value, target_value2)
        else:
            assert not (target_value == target_value2).any()

        # Check param update effect on targets
        target_value = loss_fn.target_mixer_network_params.clone()
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += 3
        target_value2 = loss_fn.target_mixer_network_params.clone()
        if loss_fn.delay_value:
            assert_allclose_td(target_value, target_value2)
        else:
            assert not (target_value == target_value2).any()

        # check that policy is updated after parameter update
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("action_spec_type", ("one_hot", "categorical"))
    def test_qmixer_state_dict(self, delay_value, device, action_spec_type):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, device=device
        )
        mixer = self._create_mock_mixer(device=device)
        loss_fn = QMixerLoss(actor, mixer, loss_function="l2", delay_value=delay_value)
        sd = loss_fn.state_dict()
        loss_fn2 = QMixerLoss(actor, mixer, loss_function="l2", delay_value=delay_value)
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("action_spec_type", ("one_hot", "categorical"))
    def test_qmix_batcher(self, n, delay_value, device, action_spec_type, gamma=0.9):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, device=device
        )
        mixer = self._create_mock_mixer(device=device)
        td = self._create_mock_data_dqn(
            action_spec_type=action_spec_type, T=4, device=device
        )
        loss_fn = QMixerLoss(actor, mixer, loss_function="l2", delay_value=delay_value)

        ms = MultiStep(gamma=gamma, n_steps=n).to(device)
        ms_td = ms(td.clone())

        with (
            pytest.warns(UserWarning, match="No target network updater has been")
            if delay_value and rl_warnings()
            else contextlib.nullcontext()
        ), _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)

        if delay_value:
            # remove warning
            SoftUpdate(loss_fn, eps=0.5)

        assert loss_fn.tensor_keys.priority in ms_td.keys()

        with torch.no_grad():
            loss = loss_fn(td)
        if n == 1:
            assert_allclose_td(td, ms_td.select(*td.keys(True, True)))
            _loss = sum(
                [item for name, item in loss.items() if name.startswith("loss")]
            )
            _loss_ms = sum(
                [item for name, item in loss_ms.items() if name.startswith("loss")]
            )
            assert (
                abs(_loss - _loss_ms) < 1e-3
            ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                assert_allclose_td(loss, loss_ms)
        sum(
            [item for name, item in loss_ms.items() if name.startswith("loss")]
        ).backward()
        assert torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 1.0) > 0.0

        # Check param update effect on targets
        target_value = loss_fn.target_local_value_network_params.clone()
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += 3
        target_value2 = loss_fn.target_local_value_network_params.clone()
        if loss_fn.delay_value:
            assert_allclose_td(target_value, target_value2)
        else:
            assert not (target_value == target_value2).any()

        # Check param update effect on targets
        target_value = loss_fn.target_mixer_network_params.clone()
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += 3
        target_value2 = loss_fn.target_mixer_network_params.clone()
        if loss_fn.delay_value:
            assert_allclose_td(target_value, target_value2)
        else:
            assert not (target_value == target_value2).any()

        # check that policy is updated after parameter update
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_qmix_tensordict_keys(self, td_est):
        torch.manual_seed(self.seed)
        action_spec_type = "one_hot"
        actor = self._create_mock_actor(action_spec_type=action_spec_type)
        mixer = self._create_mock_mixer()
        loss_fn = QMixerLoss(actor, mixer, delay_value=True)

        default_keys = {
            "advantage": "advantage",
            "value_target": "value_target",
            "local_value": ("agents", "chosen_action_value"),
            "global_value": "chosen_action_value",
            "priority": "td_error",
            "action_value": ("agents", "action_value"),
            "action": ("agents", "action"),
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
        }

        self.tensordict_keys_test(loss_fn, default_keys=default_keys)

        loss_fn = QMixerLoss(actor, mixer, delay_value=True)
        key_mapping = {
            "advantage": ("advantage", "advantage_2"),
            "value_target": ("value_target", ("value_target", "nested")),
            "reward": ("reward", "reward_test"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

        actor = self._create_mock_actor(
            action_spec_type=action_spec_type,
        )
        mixer = self._create_mock_mixer(
            global_chosen_action_value_key=("some", "nested")
        )
        loss_fn = QMixerLoss(actor, mixer, delay_value=True)
        key_mapping = {
            "global_value": ("value", ("some", "nested")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize("action_spec_type", ("categorical", "one_hot"))
    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_qmix_tensordict_run(self, action_spec_type, td_est):
        torch.manual_seed(self.seed)
        tensor_keys = {
            "action_value": ("other", "action_value_test"),
            "action": ("other", "action"),
            "local_value": ("some", "local_v"),
            "global_value": "global_v",
            "priority": "priority_test",
        }
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type,
            action_value_key=tensor_keys["action_value"],
            action_key=tensor_keys["action"],
            chosen_action_value_key=tensor_keys["local_value"],
        )
        mixer = self._create_mock_mixer(
            chosen_action_value_key=tensor_keys["local_value"],
            global_chosen_action_value_key=tensor_keys["global_value"],
        )
        td = self._create_mock_data_dqn(
            action_spec_type=action_spec_type,
            action_key=tensor_keys["action"],
            action_value_key=tensor_keys["action_value"],
        )

        loss_fn = QMixerLoss(actor, mixer, loss_function="l2", delay_value=True)
        loss_fn.set_keys(**tensor_keys)
        SoftUpdate(loss_fn, eps=0.5)
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)
        with _check_td_steady(td):
            _ = loss_fn(td)
        assert loss_fn.tensor_keys.priority in td.keys()

    @pytest.mark.parametrize(
        "mixer_local_chosen_action_value_key",
        [("agents", "chosen_action_value"), ("other")],
    )
    @pytest.mark.parametrize(
        "mixer_global_chosen_action_value_key",
        ["chosen_action_value", ("nested", "other")],
    )
    def test_mixer_keys(
        self,
        mixer_local_chosen_action_value_key,
        mixer_global_chosen_action_value_key,
        n_agents=4,
        obs_dim=3,
    ):
        torch.manual_seed(0)
        actor = self._create_mock_actor(
            action_spec_type="categorical",
        )
        mixer = self._create_mock_mixer(
            chosen_action_value_key=mixer_local_chosen_action_value_key,
            global_chosen_action_value_key=mixer_global_chosen_action_value_key,
            n_agents=n_agents,
        )

        td = TensorDict(
            {
                "agents": TensorDict(
                    {"observation": torch.zeros(32, n_agents, obs_dim)}, [32, n_agents]
                ),
                "state": torch.zeros(32, 64, 64, 3),
                "next": TensorDict(
                    {
                        "agents": TensorDict(
                            {"observation": torch.zeros(32, n_agents, obs_dim)},
                            [32, n_agents],
                        ),
                        "state": torch.zeros(32, 64, 64, 3),
                        "reward": torch.zeros(32, 1),
                        "done": torch.zeros(32, 1, dtype=torch.bool),
                        "terminated": torch.zeros(32, 1, dtype=torch.bool),
                    },
                    [32],
                ),
            },
            [32],
        )
        td = actor(td)

        loss = QMixerLoss(actor, mixer, delay_value=True)

        SoftUpdate(loss, eps=0.5)

        # Wthout etting the keys
        if mixer_local_chosen_action_value_key != ("agents", "chosen_action_value"):
            with pytest.raises(KeyError):
                loss(td)
        elif unravel_key(mixer_global_chosen_action_value_key) != "chosen_action_value":
            with pytest.raises(
                KeyError, match='key "chosen_action_value" not found in TensorDict'
            ):
                loss(td)
        else:
            loss(td)

        loss = QMixerLoss(actor, mixer, delay_value=True)

        SoftUpdate(loss, eps=0.5)

        # When setting the key
        loss.set_keys(global_value=mixer_global_chosen_action_value_key)
        if mixer_local_chosen_action_value_key != ("agents", "chosen_action_value"):
            with pytest.raises(
                KeyError
            ):  # The mixer in key still does not match the actor out_key
                loss(td)
        else:
            loss(td)

    def test_dqn_prioritized_weights(self):
        """Test DQN with prioritized replay buffer weighted loss reduction."""
        n_obs = 4
        n_actions = 3
        batch_size = 32
        buffer_size = 100

        # Create DQN value network using QValueActor
        module = nn.Linear(n_obs, n_actions)
        action_spec = Categorical(n_actions)
        value = QValueActor(
            spec=Composite(
                {
                    "action": action_spec,
                    "action_value": None,
                    "chosen_action_value": None,
                },
                shape=[],
            ),
            action_space="categorical",
            module=module,
        )

        # Create DQN loss
        loss_fn = DQNLoss(
            value_network=value, action_space="categorical", reduction="mean"
        )
        softupdate = SoftUpdate(loss_fn, eps=0.5)
        loss_fn.make_value_estimator()

        # Create prioritized replay buffer
        rb = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.9,
            storage=LazyTensorStorage(buffer_size),
            batch_size=batch_size,
            priority_key="td_error",
        )

        # Create initial data
        initial_data = TensorDict(
            {
                "observation": torch.randn(buffer_size, n_obs),
                "action": torch.randint(0, n_actions, (buffer_size,)),
                ("next", "observation"): torch.randn(buffer_size, n_obs),
                ("next", "reward"): torch.randn(buffer_size, 1),
                ("next", "done"): torch.zeros(buffer_size, 1, dtype=torch.bool),
                ("next", "terminated"): torch.zeros(buffer_size, 1, dtype=torch.bool),
            },
            batch_size=[buffer_size],
        )
        rb.extend(initial_data)

        # Sample (weights should all be identical initially)
        sample1 = rb.sample()
        assert "priority_weight" in sample1.keys()
        weights1 = sample1["priority_weight"]
        assert torch.allclose(weights1, weights1[0], atol=1e-5)

        # Run loss to get priorities
        loss_fn(sample1)
        assert "td_error" in sample1.keys()

        # Update replay buffer with new priorities
        rb.update_tensordict_priority(sample1)

        # Sample again - weights should now be non-equal
        sample2 = rb.sample()
        weights2 = sample2["priority_weight"]
        assert weights2.std() > 1e-5

        # Run loss again with varied weights
        loss_out2 = loss_fn(sample2)
        assert torch.isfinite(loss_out2["loss"])

        # Verify manual weighted average matches
        loss_fn_no_reduction = DQNLoss(
            value_network=value,
            action_space="categorical",
            reduction="none",
            use_prioritized_weights=False,
        )
        softupdate = SoftUpdate(loss_fn_no_reduction, eps=0.5)
        loss_fn_no_reduction.make_value_estimator()
        loss_fn_no_reduction.target_value_network_params = (
            loss_fn.target_value_network_params
        )

        loss_elements = loss_fn_no_reduction(sample2)["loss"]
        manual_weighted_loss = (loss_elements * weights2).sum() / weights2.sum()
        assert torch.allclose(loss_out2["loss"], manual_weighted_loss, rtol=1e-4)


@pytest.mark.skipif(
    not _has_functorch, reason=f"functorch not installed: {FUNCTORCH_ERR}"
)
class TestDDPG(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = Bounded(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        module = nn.Linear(obs_dim, action_dim)
        actor = Actor(
            spec=action_spec,
            module=module,
        )
        return actor.to(device)

    def _create_mock_value(
        self, batch=2, obs_dim=3, action_dim=4, state_dim=8, device="cpu", out_keys=None
    ):
        # Actor
        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(obs_dim + action_dim + state_dim, 1)

            def forward(self, obs, state, act):
                return self.linear(torch.cat([obs, state, act], -1))

        module = ValueClass()
        value = ValueOperator(
            module=module, in_keys=["observation", "state", "action"], out_keys=out_keys
        )
        return value.to(device)

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_common_layer_setup(
        self, n_obs=3, n_act=4, ncells=4, batch=2, n_hidden=2
    ):
        common = MLP(
            num_cells=ncells,
            in_features=n_obs,
            depth=3,
            out_features=n_hidden,
        )
        actor = MLP(
            num_cells=ncells,
            in_features=n_hidden,
            depth=1,
            out_features=n_act,
        )
        value = MLP(
            in_features=n_hidden + n_act,
            num_cells=ncells,
            depth=1,
            out_features=1,
        )
        batch = [batch]
        td = TensorDict(
            {
                "obs": torch.randn(*batch, n_obs),
                "action": torch.randn(*batch, n_act),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
                "terminated": torch.zeros(*batch, 1, dtype=torch.bool),
                "next": {
                    "obs": torch.randn(*batch, n_obs),
                    "reward": torch.randn(*batch, 1),
                    "done": torch.zeros(*batch, 1, dtype=torch.bool),
                    "terminated": torch.zeros(*batch, 1, dtype=torch.bool),
                },
            },
            batch,
        )
        common = Mod(common, in_keys=["obs"], out_keys=["hidden"])
        actor_head = Mod(actor, in_keys=["hidden"], out_keys=["action"])
        actor = Seq(common, actor_head)
        value_head = Mod(
            value, in_keys=["hidden", "action"], out_keys=["state_action_value"]
        )
        value = Seq(common, value_head)
        value(actor(td))
        return actor, value, common, td

    def _create_mock_data_ddpg(
        self,
        batch=8,
        obs_dim=3,
        action_dim=4,
        state_dim=8,
        atoms=None,
        device="cpu",
        reward_key="reward",
        done_key="done",
        terminated_key="terminated",
    ):
        # create a tensordict
        obs = torch.randn(batch, obs_dim, device=device)
        next_obs = torch.randn(batch, obs_dim, device=device)
        if atoms:
            raise NotImplementedError
        else:
            action = torch.randn(batch, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, 1, device=device)
        state = torch.randn(batch, state_dim, device=device)
        done = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "state": state,
                "next": {
                    "observation": next_obs,
                    "state": state,
                    done_key: done,
                    terminated_key: terminated,
                    reward_key: reward,
                },
                "action": action,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_ddpg(
        self,
        batch=8,
        T=4,
        obs_dim=3,
        action_dim=4,
        state_dim=8,
        atoms=None,
        device="cpu",
        reward_key="reward",
        done_key="done",
        terminated_key="terminated",
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim, device=device)
        total_state = torch.randn(batch, T + 1, state_dim, device=device)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        state = total_state[:, :T]
        next_state = total_state[:, 1:]
        if atoms:
            action = torch.randn(batch, T, atoms, action_dim, device=device).clamp(
                -1, 1
            )
        else:
            action = torch.randn(batch, T, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, T, 1, device=device)

        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = ~torch.zeros(batch, T, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "state": state.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "next": {
                    "observation": next_obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                    "state": next_state.masked_fill_(~mask.unsqueeze(-1), 0.0),
                    done_key: done,
                    terminated_key: terminated,
                    reward_key: reward.masked_fill_(~mask.unsqueeze(-1), 0.0),
                },
                "collector": {"mask": mask},
                "action": action.masked_fill_(~mask.unsqueeze(-1), 0.0),
            },
            names=[None, "time"],
            device=device,
        )
        return td

    def test_reset_parameters_recursive(self):
        actor = self._create_mock_actor()
        value = self._create_mock_value()
        loss_fn = DDPGLoss(actor, value)
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("delay_actor,delay_value", [(False, False), (True, True)])
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_ddpg(self, delay_actor, delay_value, device, td_est):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        td = self._create_mock_data_ddpg(device=device)
        loss_fn = DDPGLoss(
            actor,
            value,
            loss_function="l2",
            delay_actor=delay_actor,
            delay_value=delay_value,
        )
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)

        with _check_td_steady(td), (
            pytest.warns(UserWarning, match="No target network updater has been")
            if (delay_actor or delay_value) and rl_warnings()
            else contextlib.nullcontext()
        ):
            loss = loss_fn(td)

        if delay_value:
            # remove warning
            SoftUpdate(loss_fn, eps=0.5)

        assert all(
            (p.grad is None) or (p.grad == 0).all()
            for p in loss_fn.value_network_params.values(True, True)
        )
        assert all(
            (p.grad is None) or (p.grad == 0).all()
            for p in loss_fn.actor_network_params.values(True, True)
        )
        # check that losses are independent
        for k in loss.keys():
            if not k.startswith("loss"):
                continue
            loss[k].sum().backward(retain_graph=True)
            if k == "loss_actor":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params.values(True, True)
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(True, True)
                )
            elif k == "loss_value":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(True, True)
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params.values(True, True)
                )
            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()

        # check overall grad
        sum(
            [item for name, item in loss.items() if name.startswith("loss_")]
        ).backward()
        parameters = list(actor.parameters()) + list(value.parameters())
        for p in parameters:
            assert p.grad.norm() > 0.0

        # Check param update effect on targets
        target_actor = [p.clone() for p in loss_fn.target_actor_network_params.values()]
        target_value = [p.clone() for p in loss_fn.target_value_network_params.values()]
        _i = -1
        for _i, p in enumerate(loss_fn.parameters()):
            if p.requires_grad:
                p.data += torch.randn_like(p)
        assert _i >= 0
        target_actor2 = [
            p.clone() for p in loss_fn.target_actor_network_params.values()
        ]
        target_value2 = [
            p.clone() for p in loss_fn.target_value_network_params.values()
        ]
        if loss_fn.delay_actor:
            assert all((p1 == p2).all() for p1, p2 in zip(target_actor, target_actor2))
        else:
            assert not any(
                (p1 == p2).any() for p1, p2 in zip(target_actor, target_actor2)
            )
        if loss_fn.delay_value:
            assert all((p1 == p2).all() for p1, p2 in zip(target_value, target_value2))
        else:
            assert not any(
                (p1 == p2).any() for p1, p2 in zip(target_value, target_value2)
            )

        # check that policy is updated after parameter update
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("delay_actor,delay_value", [(False, False), (True, True)])
    def test_ddpg_state_dict(self, delay_actor, delay_value, device):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        loss_fn = DDPGLoss(
            actor,
            value,
            loss_function="l2",
            delay_actor=delay_actor,
            delay_value=delay_value,
        )
        state_dict = loss_fn.state_dict()
        loss_fn2 = DDPGLoss(
            actor,
            value,
            loss_function="l2",
            delay_actor=delay_actor,
            delay_value=delay_value,
        )
        loss_fn2.load_state_dict(state_dict)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("separate_losses", [False, True])
    def test_ddpg_separate_losses(
        self,
        device,
        separate_losses,
    ):
        torch.manual_seed(self.seed)
        actor, value, common, td = self._create_mock_common_layer_setup()
        loss_fn = DDPGLoss(
            actor,
            value,
            separate_losses=separate_losses,
        )

        with pytest.warns(
            UserWarning, match="No target network updater has been"
        ) if rl_warnings() else contextlib.nullcontext():
            loss = loss_fn(td)

        # remove warning
        SoftUpdate(loss_fn, eps=0.5)

        assert all(
            (p.grad is None) or (p.grad == 0).all()
            for p in loss_fn.value_network_params.values(True, True)
        )
        assert all(
            (p.grad is None) or (p.grad == 0).all()
            for p in loss_fn.actor_network_params.values(True, True)
        )

        # check that losses are independent
        for k in loss.keys():
            if not k.startswith("loss"):
                continue
            loss[k].sum().backward(retain_graph=True)
            common_layers_no = len(list(common.parameters()))
            if k == "loss_actor":
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(True, True)
                )
                if separate_losses:
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.value_network_params.values(True, True)
                    )
                else:
                    common_layers = itertools.islice(
                        loss_fn.value_network_params.values(True, True),
                        common_layers_no,
                    )
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all() for p in common_layers
                    )
                    value_layers = itertools.islice(
                        loss_fn.value_network_params.values(True, True),
                        common_layers_no,
                        None,
                    )
                    assert all(
                        (p.grad is None) or (p.grad == 0).all() for p in value_layers
                    )
            elif k == "loss_value":
                if separate_losses:
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(True, True)
                    )
                else:
                    if separate_losses:
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in loss_fn.actor_network_params.values(True, True)
                        )
                        common_layers = itertools.islice(
                            loss_fn.value_network_params.values(True, True),
                            common_layers_no,
                        )
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in common_layers
                        )
                        value_layers = itertools.islice(
                            loss_fn.value_network_params.values(True, True),
                            common_layers_no,
                            None,
                        )
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in value_layers
                        )
                    else:
                        common_layers = itertools.islice(
                            loss_fn.actor_network_params.values(True, True),
                            common_layers_no,
                        )
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in common_layers
                        )
                        actor_layers = itertools.islice(
                            loss_fn.actor_network_params.values(True, True),
                            common_layers_no,
                            None,
                        )
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in actor_layers
                        )
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in loss_fn.value_network_params.values(True, True)
                        )
            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()

    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("delay_actor,delay_value", [(False, False), (True, True)])
    def test_ddpg_batcher(self, n, delay_actor, delay_value, device, gamma=0.9):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        td = self._create_seq_mock_data_ddpg(device=device)
        loss_fn = DDPGLoss(
            actor,
            value,
            loss_function="l2",
            delay_actor=delay_actor,
            delay_value=delay_value,
        )

        ms = MultiStep(gamma=gamma, n_steps=n).to(device)
        ms_td = ms(td.clone())
        with (
            pytest.warns(UserWarning, match="No target network updater has been")
            if (delay_value or delay_value) and rl_warnings()
            else contextlib.nullcontext()
        ), _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)

        if delay_value:
            # remove warning
            SoftUpdate(loss_fn, eps=0.5)

        with torch.no_grad():
            loss = loss_fn(td)
        if n == 1:
            assert_allclose_td(td, ms_td.select(*list(td.keys(True, True))))
            _loss = sum(
                [item for name, item in loss.items() if name.startswith("loss_")]
            )
            _loss_ms = sum(
                [item for name, item in loss_ms.items() if name.startswith("loss_")]
            )
            assert (
                abs(_loss - _loss_ms) < 1e-3
            ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                assert_allclose_td(loss, loss_ms)
        sum(
            [item for name, item in loss_ms.items() if name.startswith("loss_")]
        ).backward()
        parameters = list(actor.parameters()) + list(value.parameters())
        for p in parameters:
            assert p.grad.norm() > 0.0

    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_ddpg_tensordict_keys(self, td_est):
        actor = self._create_mock_actor()
        value = self._create_mock_value()
        loss_fn = DDPGLoss(
            actor,
            value,
            loss_function="l2",
        )

        default_keys = {
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
            "state_action_value": "state_action_value",
            "priority": "td_error",
        }

        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
            td_est=td_est,
        )

        value = self._create_mock_value(out_keys=["state_action_value_test"])
        loss_fn = DDPGLoss(
            actor,
            value,
            loss_function="l2",
        )
        key_mapping = {
            "state_action_value": ("value", "state_action_value_test"),
            "reward": ("reward", "reward2"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize(
        "td_est",
        [ValueEstimators.TD0, ValueEstimators.TD1, ValueEstimators.TDLambda, None],
    )
    def test_ddpg_tensordict_run(self, td_est):
        """Test DDPG loss module with non-default tensordict keys."""
        torch.manual_seed(self.seed)
        tensor_keys = {
            "state_action_value": "state_action_value_test",
            "priority": "td_error_test",
            "reward": "reward_test",
            "done": ("done", "test"),
            "terminated": ("terminated", "test"),
        }

        actor = self._create_mock_actor()
        value = self._create_mock_value(out_keys=[tensor_keys["state_action_value"]])
        td = self._create_mock_data_ddpg(
            reward_key="reward_test",
            done_key=("done", "test"),
            terminated_key=("terminated", "test"),
        )
        loss_fn = DDPGLoss(
            actor,
            value,
            loss_function="l2",
        )
        loss_fn.set_keys(**tensor_keys)

        if td_est is not None:
            loss_fn.make_value_estimator(td_est)

        with _check_td_steady(td), pytest.warns(
            UserWarning, match="No target network updater has been"
        ) if rl_warnings() else contextlib.nullcontext():
            _ = loss_fn(td)

    def test_ddpg_notensordict(self):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor()
        value = self._create_mock_value()
        td = self._create_mock_data_ddpg()
        loss = DDPGLoss(actor, value)

        kwargs = {
            "observation": td.get("observation"),
            "next_reward": td.get(("next", "reward")),
            "next_done": td.get(("next", "done")),
            "next_terminated": td.get(("next", "terminated")),
            "next_observation": td.get(("next", "observation")),
            "action": td.get("action"),
            "state": td.get("state"),
            "next_state": td.get(("next", "state")),
        }
        td = TensorDict(kwargs, td.batch_size).unflatten_keys("_")

        with pytest.warns(
            UserWarning, match="No target network updater has been"
        ) if rl_warnings() else contextlib.nullcontext():
            loss_val_td = loss(td)
            loss_val = loss(**kwargs)
            for i, key in enumerate(loss.out_keys):
                torch.testing.assert_close(loss_val_td.get(key), loss_val[i])
            # test select
            loss.select_out_keys("loss_actor", "target_value")
            if torch.__version__ >= "2.0.0":
                loss_actor, target_value = loss(**kwargs)
            else:
                with pytest.raises(
                    RuntimeError,
                    match="You are likely using tensordict.nn.dispatch with keyword arguments",
                ):
                    loss_actor, target_value = loss(**kwargs)
                return
            assert loss_actor == loss_val_td["loss_actor"]
            assert (target_value == loss_val_td["target_value"]).all()

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_ddpg_reduction(self, reduction):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        td = self._create_mock_data_ddpg(device=device)
        loss_fn = DDPGLoss(
            actor,
            value,
            loss_function="l2",
            delay_actor=False,
            delay_value=False,
            reduction=reduction,
        )
        loss_fn.make_value_estimator()
        loss = loss_fn(td)
        if reduction == "none":
            for key in loss.keys():
                if key.startswith("loss"):
                    assert loss[key].shape == td.shape
        else:
            for key in loss.keys():
                if not key.startswith("loss_"):
                    continue
                assert loss[key].shape == torch.Size([])

    def test_ddpg_prioritized_weights(self):
        """Test DDPG with prioritized replay buffer weighted loss reduction."""
        n_obs = 4
        n_act = 2
        batch_size = 32
        buffer_size = 100

        # Actor network
        actor_net = MLP(in_features=n_obs, out_features=n_act, num_cells=[64, 64])
        actor = ValueOperator(
            module=actor_net,
            in_keys=["observation"],
            out_keys=["action"],
        )

        # Q-value network
        qvalue_net = MLP(in_features=n_obs + n_act, out_features=1, num_cells=[64, 64])
        qvalue = ValueOperator(module=qvalue_net, in_keys=["observation", "action"])

        # Create DDPG loss
        loss_fn = DDPGLoss(actor_network=actor, value_network=qvalue)
        softupdate = SoftUpdate(loss_fn, eps=0.5)
        loss_fn.make_value_estimator()

        # Create prioritized replay buffer
        rb = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.9,
            storage=LazyTensorStorage(buffer_size),
            batch_size=batch_size,
            priority_key="td_error",
        )

        # Create initial data
        initial_data = TensorDict(
            {
                "observation": torch.randn(buffer_size, n_obs),
                "action": torch.randn(buffer_size, n_act).clamp(-1, 1),
                ("next", "observation"): torch.randn(buffer_size, n_obs),
                ("next", "reward"): torch.randn(buffer_size, 1),
                ("next", "done"): torch.zeros(buffer_size, 1, dtype=torch.bool),
                ("next", "terminated"): torch.zeros(buffer_size, 1, dtype=torch.bool),
            },
            batch_size=[buffer_size],
        )
        rb.extend(initial_data)

        # Sample (weights should all be identical initially)
        sample1 = rb.sample()
        assert "priority_weight" in sample1.keys()
        weights1 = sample1["priority_weight"]
        assert torch.allclose(weights1, weights1[0], atol=1e-5)

        # Run loss to get priorities
        loss_fn(sample1)
        assert "td_error" in sample1.keys()

        # Update replay buffer with new priorities
        rb.update_tensordict_priority(sample1)

        # Sample again - weights should now be non-equal
        sample2 = rb.sample()
        weights2 = sample2["priority_weight"]
        assert weights2.std() > 1e-5

        # Run loss again with varied weights
        loss_out2 = loss_fn(sample2)
        assert torch.isfinite(loss_out2["loss_value"])

        # Verify weighted vs unweighted differ
        loss_fn_no_weights = DDPGLoss(
            actor_network=actor,
            value_network=qvalue,
            use_prioritized_weights=False,
        )
        softupdate = SoftUpdate(loss_fn_no_weights, eps=0.5)
        loss_fn_no_weights.make_value_estimator()
        loss_fn_no_weights.value_network_params = loss_fn.value_network_params
        loss_fn_no_weights.target_value_network_params = (
            loss_fn.target_value_network_params
        )
        loss_fn_no_weights.actor_network_params = loss_fn.actor_network_params
        loss_fn_no_weights.target_actor_network_params = (
            loss_fn.target_actor_network_params
        )

        loss_out_no_weights = loss_fn_no_weights(sample2)
        # Weighted and unweighted should differ (in general)
        assert torch.isfinite(loss_out_no_weights["loss_value"])


@pytest.mark.skipif(
    not _has_functorch, reason=f"functorch not installed: {FUNCTORCH_ERR}"
)
class TestTD3(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        in_keys=None,
        out_keys=None,
        dropout=0.0,
    ):
        # Actor
        action_spec = Bounded(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        module = nn.Sequential(
            nn.Linear(obs_dim, obs_dim),
            nn.Dropout(dropout),
            nn.Linear(obs_dim, action_dim),
        )
        actor = Actor(
            spec=action_spec, module=module, in_keys=in_keys, out_keys=out_keys
        )
        return actor.to(device)

    def _create_mock_value(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        out_keys=None,
        action_key="action",
        observation_key="observation",
    ):
        # Actor
        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(obs_dim + action_dim, 1)

            def forward(self, obs, act):
                return self.linear(torch.cat([obs, act], -1))

        module = ValueClass()
        value = ValueOperator(
            module=module,
            in_keys=[observation_key, action_key],
            out_keys=out_keys,
        )
        return value.to(device)

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_common_layer_setup(
        self, n_obs=3, n_act=4, ncells=4, batch=2, n_hidden=2
    ):
        common = MLP(
            num_cells=ncells,
            in_features=n_obs,
            depth=3,
            out_features=n_hidden,
        )
        actor_net = MLP(
            num_cells=ncells,
            in_features=n_hidden,
            depth=1,
            out_features=2 * n_act,
        )
        value = MLP(
            in_features=n_hidden + n_act,
            num_cells=ncells,
            depth=1,
            out_features=1,
        )
        batch = [batch]
        td = TensorDict(
            {
                "obs": torch.randn(*batch, n_obs),
                "action": torch.randn(*batch, n_act),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
                "terminated": torch.zeros(*batch, 1, dtype=torch.bool),
                "next": {
                    "obs": torch.randn(*batch, n_obs),
                    "reward": torch.randn(*batch, 1),
                    "done": torch.zeros(*batch, 1, dtype=torch.bool),
                    "terminated": torch.zeros(*batch, 1, dtype=torch.bool),
                },
            },
            batch,
        )
        common = Mod(common, in_keys=["obs"], out_keys=["hidden"])
        actor = ProbSeq(
            common,
            Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
            Mod(NormalParamExtractor(), in_keys=["param"], out_keys=["loc", "scale"]),
            ProbMod(
                in_keys=["loc", "scale"],
                out_keys=["action"],
                distribution_class=TanhNormal,
                return_log_prob=True,
            ),
        )
        value_head = Mod(
            value, in_keys=["hidden", "action"], out_keys=["state_action_value"]
        )
        value = Seq(common, value_head)
        return actor, value, common, td

    def _create_mock_data_td3(
        self,
        batch=8,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
        action_key="action",
        observation_key="observation",
        reward_key="reward",
        done_key="done",
        terminated_key="terminated",
    ):
        # create a tensordict
        obs = torch.randn(batch, obs_dim, device=device)
        next_obs = torch.randn(batch, obs_dim, device=device)
        if atoms:
            raise NotImplementedError
        else:
            action = torch.randn(batch, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, 1, device=device)
        done = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch,),
            source={
                observation_key: obs,
                "next": {
                    observation_key: next_obs,
                    done_key: done,
                    terminated_key: terminated,
                    reward_key: reward,
                },
                action_key: action,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_td3(
        self, batch=8, T=4, obs_dim=3, action_dim=4, atoms=None, device="cpu"
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim, device=device)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action = torch.randn(batch, T, atoms, action_dim, device=device).clamp(
                -1, 1
            )
        else:
            action = torch.randn(batch, T, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = ~torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs * mask.to(obs.dtype),
                "next": {
                    "observation": next_obs * mask.to(obs.dtype),
                    "reward": reward * mask.to(obs.dtype),
                    "done": done,
                    "terminated": terminated,
                },
                "collector": {"mask": mask},
                "action": action * mask.to(obs.dtype),
            },
            names=[None, "time"],
            device=device,
        )
        return td

    def test_reset_parameters_recursive(self):
        actor = self._create_mock_actor()
        value = self._create_mock_value()
        loss_fn = TD3Loss(
            actor,
            value,
            bounds=(-1, 1),
        )
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize(
        "delay_actor, delay_qvalue", [(False, False), (True, True)]
    )
    @pytest.mark.parametrize("policy_noise", [0.1, 1.0])
    @pytest.mark.parametrize("noise_clip", [0.1, 1.0])
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    @pytest.mark.parametrize("use_action_spec", [True, False])
    @pytest.mark.parametrize("dropout", [0.0, 0.1])
    def test_td3(
        self,
        delay_actor,
        delay_qvalue,
        device,
        policy_noise,
        noise_clip,
        td_est,
        use_action_spec,
        dropout,
    ):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device, dropout=dropout)
        value = self._create_mock_value(device=device)
        td = self._create_mock_data_td3(device=device)
        if use_action_spec:
            action_spec = actor.spec
            bounds = None
        else:
            bounds = (-1, 1)
            action_spec = None
        loss_fn = TD3Loss(
            actor,
            value,
            action_spec=action_spec,
            bounds=bounds,
            loss_function="l2",
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
        )
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)
        with (
            pytest.warns(
                UserWarning,
                match="No target network updater has been associated with this loss module",
            )
            if (delay_actor or delay_qvalue) and rl_warnings()
            else contextlib.nullcontext()
        ):
            with _check_td_steady(td):
                loss = loss_fn(td)

            assert all(
                (p.grad is None) or (p.grad == 0).all()
                for p in loss_fn.qvalue_network_params.values(True, True)
            )
            assert all(
                (p.grad is None) or (p.grad == 0).all()
                for p in loss_fn.actor_network_params.values(True, True)
            )
            # check that losses are independent
            for k in loss.keys():
                if not k.startswith("loss"):
                    continue
                loss[k].sum().backward(retain_graph=True)
                if k == "loss_actor":
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.qvalue_network_params.values(True, True)
                    )
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(True, True)
                    )
                elif k == "loss_qvalue":
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(True, True)
                    )
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.qvalue_network_params.values(True, True)
                    )
                else:
                    raise NotImplementedError(k)
                loss_fn.zero_grad()

            sum(
                [item for name, item in loss.items() if name.startswith("loss_")]
            ).backward()
            named_parameters = list(loss_fn.named_parameters())
            named_buffers = list(loss_fn.named_buffers())

            assert len({p for n, p in named_parameters}) == len(list(named_parameters))
            assert len({p for n, p in named_buffers}) == len(list(named_buffers))

            for name, p in named_parameters:
                if not name.startswith("target_"):
                    assert (
                        p.grad is not None and p.grad.norm() > 0.0
                    ), f"parameter {name} (shape: {p.shape}) has a null gradient"
                else:
                    assert (
                        p.grad is None or p.grad.norm() == 0.0
                    ), f"target parameter {name} (shape: {p.shape}) has a non-null gradient"

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("delay_actor, delay_qvalue", [(True, True)])
    @pytest.mark.parametrize("policy_noise", [0.1])
    @pytest.mark.parametrize("noise_clip", [0.1])
    @pytest.mark.parametrize("td_est", [None])
    @pytest.mark.parametrize("use_action_spec", [True])
    @pytest.mark.parametrize("dropout", [0.0])
    def test_td3_deactivate_vmap(
        self,
        delay_actor,
        delay_qvalue,
        device,
        policy_noise,
        noise_clip,
        td_est,
        use_action_spec,
        dropout,
    ):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device, dropout=dropout)
        value = self._create_mock_value(device=device)
        td = self._create_mock_data_td3(device=device)
        if use_action_spec:
            action_spec = actor.spec
            bounds = None
        else:
            bounds = (-1, 1)
            action_spec = None
        torch.manual_seed(0)
        loss_fn_vmap = TD3Loss(
            actor,
            value,
            action_spec=action_spec,
            bounds=bounds,
            loss_function="l2",
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
        )
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn_vmap.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn_vmap.make_value_estimator(td_est)
        tdc = td.clone()
        with (
            pytest.warns(
                UserWarning,
                match="No target network updater has been associated with this loss module",
            )
            if (delay_actor or delay_qvalue) and rl_warnings()
            else contextlib.nullcontext()
        ), _check_td_steady(td):
            torch.manual_seed(1)
            loss_vmap = loss_fn_vmap(td)
        td = tdc
        torch.manual_seed(0)
        loss_fn_no_vmap = TD3Loss(
            actor,
            value,
            action_spec=action_spec,
            bounds=bounds,
            loss_function="l2",
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
        )
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn_no_vmap.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn_no_vmap.make_value_estimator(td_est)
        with (
            pytest.warns(
                UserWarning,
                match="No target network updater has been associated with this loss module",
            )
            if (delay_actor or delay_qvalue) and rl_warnings()
            else contextlib.nullcontext()
        ), _check_td_steady(td):
            torch.manual_seed(1)
            loss_no_vmap = loss_fn_no_vmap(td)
        assert_allclose_td(loss_vmap, loss_no_vmap)

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize(
        "delay_actor, delay_qvalue", [(False, False), (True, True)]
    )
    @pytest.mark.parametrize("policy_noise", [0.1])
    @pytest.mark.parametrize("noise_clip", [0.1])
    @pytest.mark.parametrize("use_action_spec", [True, False])
    def test_td3_state_dict(
        self,
        delay_actor,
        delay_qvalue,
        device,
        policy_noise,
        noise_clip,
        use_action_spec,
    ):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        if use_action_spec:
            action_spec = actor.spec
            bounds = None
        else:
            bounds = (-1, 1)
            action_spec = None
        loss_fn = TD3Loss(
            actor,
            value,
            action_spec=action_spec,
            bounds=bounds,
            loss_function="l2",
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
        )
        sd = loss_fn.state_dict()
        loss_fn2 = TD3Loss(
            actor,
            value,
            action_spec=action_spec,
            bounds=bounds,
            loss_function="l2",
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
        )
        loss_fn2.load_state_dict(sd)

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("separate_losses", [False, True])
    def test_td3_separate_losses(
        self,
        device,
        separate_losses,
        n_act=4,
    ):
        torch.manual_seed(self.seed)
        actor, value, common, td = self._create_mock_common_layer_setup(n_act=n_act)
        loss_fn = TD3Loss(
            actor,
            value,
            action_spec=Bounded(shape=(n_act,), low=-1, high=1),
            loss_function="l2",
            separate_losses=separate_losses,
        )
        with pytest.warns(
            UserWarning, match="No target network updater has been"
        ) if rl_warnings() else contextlib.nullcontext():
            loss = loss_fn(td)

            assert all(
                (p.grad is None) or (p.grad == 0).all()
                for p in loss_fn.qvalue_network_params.values(True, True)
            )
            assert all(
                (p.grad is None) or (p.grad == 0).all()
                for p in loss_fn.actor_network_params.values(True, True)
            )
            # check that losses are independent
            for k in loss.keys():
                if not k.startswith("loss"):
                    continue
                loss[k].sum().backward(retain_graph=True)
                if k == "loss_actor":
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.qvalue_network_params.values(True, True)
                    )
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(True, True)
                    )
                elif k == "loss_qvalue":
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(True, True)
                    )
                    if separate_losses:
                        common_layers_no = len(list(common.parameters()))
                        common_layers = itertools.islice(
                            loss_fn.qvalue_network_params.values(True, True),
                            common_layers_no,
                        )
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in common_layers
                        )
                        qvalue_layers = itertools.islice(
                            loss_fn.qvalue_network_params.values(True, True),
                            common_layers_no,
                            None,
                        )
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in qvalue_layers
                        )
                    else:
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in loss_fn.qvalue_network_params.values(True, True)
                        )

                else:
                    raise NotImplementedError(k)
                loss_fn.zero_grad()

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("delay_actor,delay_qvalue", [(False, False), (True, True)])
    @pytest.mark.parametrize("policy_noise", [0.1, 1.0])
    @pytest.mark.parametrize("noise_clip", [0.1, 1.0])
    def test_td3_batcher(
        self, n, delay_actor, delay_qvalue, device, policy_noise, noise_clip, gamma=0.9
    ):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        td = self._create_seq_mock_data_td3(device=device)
        loss_fn = TD3Loss(
            actor,
            value,
            action_spec=actor.spec,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            delay_qvalue=delay_qvalue,
            delay_actor=delay_actor,
        )

        ms = MultiStep(gamma=gamma, n_steps=n).to(device)

        td_clone = td.clone()
        ms_td = ms(td_clone)

        torch.manual_seed(0)
        np.random.seed(0)

        with (
            pytest.warns(UserWarning, match="No target network updater has been")
            if (delay_qvalue or delay_actor) and rl_warnings()
            else contextlib.nullcontext()
        ), _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)
        assert loss_fn.tensor_keys.priority in ms_td.keys()

        if delay_qvalue or delay_actor:
            SoftUpdate(loss_fn, eps=0.5)

        with torch.no_grad():
            torch.manual_seed(0)  # log-prob is computed with a random action
            np.random.seed(0)
            loss = loss_fn(td)

        if n == 1:
            assert_allclose_td(td, ms_td.select(*list(td.keys(True, True))))
            _loss = sum(
                [item for name, item in loss.items() if name.startswith("loss_")]
            )
            _loss_ms = sum(
                [item for name, item in loss_ms.items() if name.startswith("loss_")]
            )
            assert (
                abs(_loss - _loss_ms) < 1e-3
            ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                assert_allclose_td(loss, loss_ms)

        sum(
            [item for name, item in loss_ms.items() if name.startswith("loss_")]
        ).backward()
        named_parameters = loss_fn.named_parameters()

        for name, p in named_parameters:
            if not name.startswith("target_"):
                assert (
                    p.grad is not None and p.grad.norm() > 0.0
                ), f"parameter {name} (shape: {p.shape}) has a null gradient"
            else:
                assert (
                    p.grad is None or p.grad.norm() == 0.0
                ), f"target parameter {name} (shape: {p.shape}) has a non-null gradient"

        # Check param update effect on targets
        target_actor = loss_fn.target_actor_network_params.clone().values(
            include_nested=True, leaves_only=True
        )
        target_qvalue = loss_fn.target_qvalue_network_params.clone().values(
            include_nested=True, leaves_only=True
        )
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        target_actor2 = loss_fn.target_actor_network_params.clone().values(
            include_nested=True, leaves_only=True
        )
        target_qvalue2 = loss_fn.target_qvalue_network_params.clone().values(
            include_nested=True, leaves_only=True
        )
        if loss_fn.delay_actor:
            assert all((p1 == p2).all() for p1, p2 in zip(target_actor, target_actor2))
        else:
            assert not any(
                (p1 == p2).any() for p1, p2 in zip(target_actor, target_actor2)
            )
        if loss_fn.delay_qvalue:
            assert all(
                (p1 == p2).all() for p1, p2 in zip(target_qvalue, target_qvalue2)
            )
        else:
            assert not any(
                (p1 == p2).any() for p1, p2 in zip(target_qvalue, target_qvalue2)
            )

        # check that policy is updated after parameter update
        actorp_set = set(actor.parameters())
        loss_fnp_set = set(loss_fn.parameters())
        assert len(actorp_set.intersection(loss_fnp_set)) == len(actorp_set)
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_td3_tensordict_keys(self, td_est):
        actor = self._create_mock_actor()
        value = self._create_mock_value()
        loss_fn = TD3Loss(
            actor,
            value,
            action_spec=actor.spec,
        )

        default_keys = {
            "priority": "td_error",
            "state_action_value": "state_action_value",
            "action": "action",
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
        }

        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
            td_est=td_est,
        )

        value = self._create_mock_value(out_keys=["state_action_value_test"])
        loss_fn = TD3Loss(
            actor,
            value,
            action_spec=actor.spec,
        )
        key_mapping = {
            "state_action_value": ("value", "state_action_value_test"),
            "reward": ("reward", "reward_test"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize("spec", [True, False])
    @pytest.mark.parametrize("bounds", [True, False])
    def test_constructor(self, spec, bounds):
        actor = self._create_mock_actor()
        value = self._create_mock_value()
        action_spec = actor.spec if spec else None
        bounds = (-1, 1) if bounds else None
        if (bounds is not None and action_spec is not None) or (
            bounds is None and action_spec is None
        ):
            with pytest.raises(ValueError, match="but not both"):
                TD3Loss(
                    actor,
                    value,
                    action_spec=action_spec,
                    bounds=bounds,
                )
            return
        TD3Loss(
            actor,
            value,
            action_spec=action_spec,
            bounds=bounds,
        )

    # TODO: test for action_key, atm the action key of the TD3 loss is not configurable,
    # since it is used in it's constructor
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    def test_td3_notensordict(
        self, observation_key, reward_key, done_key, terminated_key
    ):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(in_keys=[observation_key])
        qvalue = self._create_mock_value(
            observation_key=observation_key, out_keys=["state_action_value"]
        )
        td = self._create_mock_data_td3(
            observation_key=observation_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
        )
        loss = TD3Loss(actor, qvalue, action_spec=actor.spec)
        loss.set_keys(reward=reward_key, done=done_key, terminated=terminated_key)

        kwargs = {
            observation_key: td.get(observation_key),
            f"next_{reward_key}": td.get(("next", reward_key)),
            f"next_{done_key}": td.get(("next", done_key)),
            f"next_{terminated_key}": td.get(("next", terminated_key)),
            f"next_{observation_key}": td.get(("next", observation_key)),
            "action": td.get("action"),
        }
        td = TensorDict(kwargs, td.batch_size).unflatten_keys("_")

        with pytest.warns(
            UserWarning, match="No target network updater has been"
        ) if rl_warnings() else contextlib.nullcontext():
            torch.manual_seed(0)
            loss_val_td = loss(td)
            torch.manual_seed(0)
            loss_val = loss(**kwargs)
            loss_val_reconstruct = TensorDict(dict(zip(loss.out_keys, loss_val)), [])
            assert_allclose_td(loss_val_reconstruct, loss_val_td)

            # test select
            loss.select_out_keys("loss_actor", "loss_qvalue")
            torch.manual_seed(0)
            if torch.__version__ >= "2.0.0":
                loss_actor, loss_qvalue = loss(**kwargs)
            else:
                with pytest.raises(
                    RuntimeError,
                    match="You are likely using tensordict.nn.dispatch with keyword arguments",
                ):
                    loss_actor, loss_qvalue = loss(**kwargs)
                return

            assert loss_actor == loss_val_td["loss_actor"]
            assert loss_qvalue == loss_val_td["loss_qvalue"]

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_td3_reduction(self, reduction):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        td = self._create_mock_data_td3(device=device)
        action_spec = actor.spec
        bounds = None
        loss_fn = TD3Loss(
            actor,
            value,
            action_spec=action_spec,
            bounds=bounds,
            loss_function="l2",
            delay_qvalue=False,
            delay_actor=False,
            reduction=reduction,
        )
        loss_fn.make_value_estimator()
        loss = loss_fn(td)
        if reduction == "none":
            for key in loss.keys():
                if key.startswith("loss"):
                    assert loss[key].shape == td.shape
        else:
            for key in loss.keys():
                if not key.startswith("loss"):
                    continue
                assert loss[key].shape == torch.Size([])

    def test_td3_prioritized_weights(self):
        """Test TD3 with prioritized replay buffer weighted loss reduction."""
        n_obs = 4
        n_act = 2
        batch_size = 32
        buffer_size = 100

        # Actor network
        actor_net = MLP(in_features=n_obs, out_features=n_act, num_cells=[64, 64])
        actor = ValueOperator(
            module=actor_net,
            in_keys=["observation"],
            out_keys=["action"],
        )

        # Q-value network
        qvalue_net = MLP(in_features=n_obs + n_act, out_features=1, num_cells=[64, 64])
        qvalue = ValueOperator(module=qvalue_net, in_keys=["observation", "action"])

        # Create TD3 loss
        loss_fn = TD3Loss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=2,
            action_spec=Bounded(
                low=-torch.ones(n_act), high=torch.ones(n_act), shape=(n_act,)
            ),
        )
        softupdate = SoftUpdate(loss_fn, eps=0.5)
        loss_fn.make_value_estimator()

        # Create prioritized replay buffer
        rb = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.9,
            storage=LazyTensorStorage(buffer_size),
            batch_size=batch_size,
            priority_key="td_error",
        )

        # Create initial data
        initial_data = TensorDict(
            {
                "observation": torch.randn(buffer_size, n_obs),
                "action": torch.randn(buffer_size, n_act).clamp(-1, 1),
                ("next", "observation"): torch.randn(buffer_size, n_obs),
                ("next", "reward"): torch.randn(buffer_size, 1),
                ("next", "done"): torch.zeros(buffer_size, 1, dtype=torch.bool),
                ("next", "terminated"): torch.zeros(buffer_size, 1, dtype=torch.bool),
            },
            batch_size=[buffer_size],
        )
        rb.extend(initial_data)

        # Sample (weights should all be identical initially)
        sample1 = rb.sample()
        assert "priority_weight" in sample1.keys()
        weights1 = sample1["priority_weight"]
        assert torch.allclose(weights1, weights1[0], atol=1e-5)

        # Run loss to get priorities
        loss_fn(sample1)
        assert "td_error" in sample1.keys()

        # Update replay buffer with new priorities
        rb.update_tensordict_priority(sample1)

        # Sample again - weights should now be non-equal
        sample2 = rb.sample()
        weights2 = sample2["priority_weight"]
        assert weights2.std() > 1e-5

        # Run loss again with varied weights
        loss_out2 = loss_fn(sample2)
        assert torch.isfinite(loss_out2["loss_qvalue"])

        # Verify weighted vs unweighted differ
        loss_fn_no_weights = TD3Loss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=2,
            action_spec=Bounded(
                low=-torch.ones(n_act), high=torch.ones(n_act), shape=(n_act,)
            ),
            use_prioritized_weights=False,
        )
        softupdate = SoftUpdate(loss_fn_no_weights, eps=0.5)
        loss_fn_no_weights.make_value_estimator()
        loss_fn_no_weights.qvalue_network_params = loss_fn.qvalue_network_params
        loss_fn_no_weights.target_qvalue_network_params = (
            loss_fn.target_qvalue_network_params
        )
        loss_fn_no_weights.actor_network_params = loss_fn.actor_network_params
        loss_fn_no_weights.target_actor_network_params = (
            loss_fn.target_actor_network_params
        )

        loss_out_no_weights = loss_fn_no_weights(sample2)
        # Weighted and unweighted should differ (in general)
        assert torch.isfinite(loss_out_no_weights["loss_qvalue"])


class TestTD3BC(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        in_keys=None,
        out_keys=None,
        dropout=0.0,
    ):
        # Actor
        action_spec = Bounded(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        module = nn.Sequential(
            nn.Linear(obs_dim, obs_dim),
            nn.Dropout(dropout),
            nn.Linear(obs_dim, action_dim),
        )
        actor = Actor(
            spec=action_spec, module=module, in_keys=in_keys, out_keys=out_keys
        )
        return actor.to(device)

    def _create_mock_value(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        out_keys=None,
        action_key="action",
        observation_key="observation",
    ):
        # Actor
        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(obs_dim + action_dim, 1)

            def forward(self, obs, act):
                return self.linear(torch.cat([obs, act], -1))

        module = ValueClass()
        value = ValueOperator(
            module=module,
            in_keys=[observation_key, action_key],
            out_keys=out_keys,
        )
        return value.to(device)

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_common_layer_setup(
        self, n_obs=3, n_act=4, ncells=4, batch=2, n_hidden=2
    ):
        common = MLP(
            num_cells=ncells,
            in_features=n_obs,
            depth=3,
            out_features=n_hidden,
        )
        actor_net = MLP(
            num_cells=ncells,
            in_features=n_hidden,
            depth=1,
            out_features=2 * n_act,
        )
        value = MLP(
            in_features=n_hidden + n_act,
            num_cells=ncells,
            depth=1,
            out_features=1,
        )
        batch = [batch]
        td = TensorDict(
            {
                "obs": torch.randn(*batch, n_obs),
                "action": torch.randn(*batch, n_act),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
                "terminated": torch.zeros(*batch, 1, dtype=torch.bool),
                "next": {
                    "obs": torch.randn(*batch, n_obs),
                    "reward": torch.randn(*batch, 1),
                    "done": torch.zeros(*batch, 1, dtype=torch.bool),
                    "terminated": torch.zeros(*batch, 1, dtype=torch.bool),
                },
            },
            batch,
        )
        common = Mod(common, in_keys=["obs"], out_keys=["hidden"])
        actor = ProbSeq(
            common,
            Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
            Mod(NormalParamExtractor(), in_keys=["param"], out_keys=["loc", "scale"]),
            ProbMod(
                in_keys=["loc", "scale"],
                out_keys=["action"],
                distribution_class=TanhNormal,
                return_log_prob=True,
            ),
        )
        value_head = Mod(
            value, in_keys=["hidden", "action"], out_keys=["state_action_value"]
        )
        value = Seq(common, value_head)
        return actor, value, common, td

    def _create_mock_data_td3bc(
        self,
        batch=8,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
        action_key="action",
        observation_key="observation",
        reward_key="reward",
        done_key="done",
        terminated_key="terminated",
    ):
        # create a tensordict
        obs = torch.randn(batch, obs_dim, device=device)
        next_obs = torch.randn(batch, obs_dim, device=device)
        if atoms:
            raise NotImplementedError
        else:
            action = torch.randn(batch, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, 1, device=device)
        done = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch,),
            source={
                observation_key: obs,
                "next": {
                    observation_key: next_obs,
                    done_key: done,
                    terminated_key: terminated,
                    reward_key: reward,
                },
                action_key: action,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_td3bc(
        self, batch=8, T=4, obs_dim=3, action_dim=4, atoms=None, device="cpu"
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim, device=device)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action = torch.randn(batch, T, atoms, action_dim, device=device).clamp(
                -1, 1
            )
        else:
            action = torch.randn(batch, T, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = ~torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs * mask.to(obs.dtype),
                "next": {
                    "observation": next_obs * mask.to(obs.dtype),
                    "reward": reward * mask.to(obs.dtype),
                    "done": done,
                    "terminated": terminated,
                },
                "collector": {"mask": mask},
                "action": action * mask.to(obs.dtype),
            },
            names=[None, "time"],
            device=device,
        )
        return td

    def test_reset_parameters_recursive(self):
        actor = self._create_mock_actor()
        value = self._create_mock_value()
        loss_fn = TD3BCLoss(
            actor,
            value,
            bounds=(-1, 1),
        )
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize(
        "delay_actor, delay_qvalue", [(False, False), (True, True)]
    )
    @pytest.mark.parametrize("policy_noise", [0.1, 1.0])
    @pytest.mark.parametrize("noise_clip", [0.1, 1.0])
    @pytest.mark.parametrize("alpha", [0.1, 6.0])
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    @pytest.mark.parametrize("use_action_spec", [True, False])
    @pytest.mark.parametrize("dropout", [0.0, 0.1])
    def test_td3bc(
        self,
        delay_actor,
        delay_qvalue,
        device,
        policy_noise,
        noise_clip,
        alpha,
        td_est,
        use_action_spec,
        dropout,
    ):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device, dropout=dropout)
        value = self._create_mock_value(device=device)
        td = self._create_mock_data_td3bc(device=device)
        if use_action_spec:
            action_spec = actor.spec
            bounds = None
        else:
            bounds = (-1, 1)
            action_spec = None
        loss_fn = TD3BCLoss(
            actor,
            value,
            action_spec=action_spec,
            bounds=bounds,
            loss_function="l2",
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            alpha=alpha,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
        )
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)
        with (
            pytest.warns(
                UserWarning,
                match="No target network updater has been associated with this loss module",
            )
            if (delay_actor or delay_qvalue) and rl_warnings()
            else contextlib.nullcontext()
        ):
            with _check_td_steady(td):
                loss = loss_fn(td)

            assert all(
                (p.grad is None) or (p.grad == 0).all()
                for p in loss_fn.qvalue_network_params.values(True, True)
            )
            assert all(
                (p.grad is None) or (p.grad == 0).all()
                for p in loss_fn.actor_network_params.values(True, True)
            )
            # check that losses are independent
            for k in loss.keys():
                if not k.startswith("loss"):
                    continue
                loss[k].sum().backward(retain_graph=True)
                if k == "loss_actor":
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.qvalue_network_params.values(True, True)
                    )
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(True, True)
                    )
                elif k == "loss_qvalue":
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(True, True)
                    )
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.qvalue_network_params.values(True, True)
                    )
                else:
                    raise NotImplementedError(k)
                loss_fn.zero_grad()

            sum(
                [item for name, item in loss.items() if name.startswith("loss_")]
            ).backward()
            named_parameters = list(loss_fn.named_parameters())
            named_buffers = list(loss_fn.named_buffers())

            assert len({p for n, p in named_parameters}) == len(list(named_parameters))
            assert len({p for n, p in named_buffers}) == len(list(named_buffers))

            for name, p in named_parameters:
                if not name.startswith("target_"):
                    assert (
                        p.grad is not None and p.grad.norm() > 0.0
                    ), f"parameter {name} (shape: {p.shape}) has a null gradient"
                else:
                    assert (
                        p.grad is None or p.grad.norm() == 0.0
                    ), f"target parameter {name} (shape: {p.shape}) has a non-null gradient"

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize(
        "delay_actor, delay_qvalue", [(False, False), (True, True)]
    )
    @pytest.mark.parametrize("policy_noise", [0.1])
    @pytest.mark.parametrize("noise_clip", [0.1])
    @pytest.mark.parametrize("alpha", [0.1])
    @pytest.mark.parametrize("use_action_spec", [True, False])
    def test_td3bc_state_dict(
        self,
        delay_actor,
        delay_qvalue,
        device,
        policy_noise,
        noise_clip,
        alpha,
        use_action_spec,
    ):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        if use_action_spec:
            action_spec = actor.spec
            bounds = None
        else:
            bounds = (-1, 1)
            action_spec = None
        loss_fn = TD3BCLoss(
            actor,
            value,
            action_spec=action_spec,
            bounds=bounds,
            loss_function="l2",
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            alpha=alpha,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
        )
        sd = loss_fn.state_dict()
        loss_fn2 = TD3BCLoss(
            actor,
            value,
            action_spec=action_spec,
            bounds=bounds,
            loss_function="l2",
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            alpha=alpha,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
        )
        loss_fn2.load_state_dict(sd)

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("separate_losses", [False, True])
    def test_td3bc_separate_losses(
        self,
        device,
        separate_losses,
        n_act=4,
    ):
        torch.manual_seed(self.seed)
        actor, value, common, td = self._create_mock_common_layer_setup(n_act=n_act)
        loss_fn = TD3BCLoss(
            actor,
            value,
            action_spec=Bounded(shape=(n_act,), low=-1, high=1),
            loss_function="l2",
            separate_losses=separate_losses,
        )
        with pytest.warns(
            UserWarning, match="No target network updater has been"
        ) if rl_warnings() else contextlib.nullcontext():
            loss = loss_fn(td)

            assert all(
                (p.grad is None) or (p.grad == 0).all()
                for p in loss_fn.qvalue_network_params.values(True, True)
            )
            assert all(
                (p.grad is None) or (p.grad == 0).all()
                for p in loss_fn.actor_network_params.values(True, True)
            )
            # check that losses are independent
            for k in loss.keys():
                if not k.startswith("loss"):
                    continue
                loss[k].sum().backward(retain_graph=True)
                if k == "loss_actor":
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.qvalue_network_params.values(True, True)
                    )
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(True, True)
                    )
                elif k == "loss_qvalue":
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(True, True)
                    )
                    if separate_losses:
                        common_layers_no = len(list(common.parameters()))
                        common_layers = itertools.islice(
                            loss_fn.qvalue_network_params.values(True, True),
                            common_layers_no,
                        )
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in common_layers
                        )
                        qvalue_layers = itertools.islice(
                            loss_fn.qvalue_network_params.values(True, True),
                            common_layers_no,
                            None,
                        )
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in qvalue_layers
                        )
                    else:
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in loss_fn.qvalue_network_params.values(True, True)
                        )

                else:
                    raise NotImplementedError(k)
                loss_fn.zero_grad()

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("delay_actor,delay_qvalue", [(False, False), (True, True)])
    @pytest.mark.parametrize("policy_noise", [0.1, 1.0])
    @pytest.mark.parametrize("noise_clip", [0.1, 1.0])
    @pytest.mark.parametrize("alpha", [0.1, 6.0])
    def test_td3bc_batcher(
        self,
        n,
        delay_actor,
        delay_qvalue,
        device,
        policy_noise,
        noise_clip,
        alpha,
        gamma=0.9,
    ):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        td = self._create_seq_mock_data_td3bc(device=device)
        loss_fn = TD3BCLoss(
            actor,
            value,
            action_spec=actor.spec,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            alpha=alpha,
            delay_qvalue=delay_qvalue,
            delay_actor=delay_actor,
        )

        ms = MultiStep(gamma=gamma, n_steps=n).to(device)

        td_clone = td.clone()
        ms_td = ms(td_clone)

        torch.manual_seed(0)
        np.random.seed(0)

        with (
            pytest.warns(UserWarning, match="No target network updater has been")
            if (delay_qvalue or delay_actor) and rl_warnings()
            else contextlib.nullcontext()
        ), _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)
        assert loss_fn.tensor_keys.priority in ms_td.keys()

        if delay_qvalue or delay_actor:
            SoftUpdate(loss_fn, eps=0.5)

        with torch.no_grad():
            torch.manual_seed(0)  # log-prob is computed with a random action
            np.random.seed(0)
            loss = loss_fn(td)

        if n == 1:
            assert_allclose_td(td, ms_td.select(*list(td.keys(True, True))))
            _loss = sum(
                [item for name, item in loss.items() if name.startswith("loss_")]
            )
            _loss_ms = sum(
                [item for name, item in loss_ms.items() if name.startswith("loss_")]
            )
            assert (
                abs(_loss - _loss_ms) < 1e-3
            ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                assert_allclose_td(loss, loss_ms)

        sum(
            [item for name, item in loss_ms.items() if name.startswith("loss_")]
        ).backward()
        named_parameters = loss_fn.named_parameters()

        for name, p in named_parameters:
            if not name.startswith("target_"):
                assert (
                    p.grad is not None and p.grad.norm() > 0.0
                ), f"parameter {name} (shape: {p.shape}) has a null gradient"
            else:
                assert (
                    p.grad is None or p.grad.norm() == 0.0
                ), f"target parameter {name} (shape: {p.shape}) has a non-null gradient"

        # Check param update effect on targets
        target_actor = loss_fn.target_actor_network_params.clone().values(
            include_nested=True, leaves_only=True
        )
        target_qvalue = loss_fn.target_qvalue_network_params.clone().values(
            include_nested=True, leaves_only=True
        )
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        target_actor2 = loss_fn.target_actor_network_params.clone().values(
            include_nested=True, leaves_only=True
        )
        target_qvalue2 = loss_fn.target_qvalue_network_params.clone().values(
            include_nested=True, leaves_only=True
        )
        if loss_fn.delay_actor:
            assert all((p1 == p2).all() for p1, p2 in zip(target_actor, target_actor2))
        else:
            assert not any(
                (p1 == p2).any() for p1, p2 in zip(target_actor, target_actor2)
            )
        if loss_fn.delay_qvalue:
            assert all(
                (p1 == p2).all() for p1, p2 in zip(target_qvalue, target_qvalue2)
            )
        else:
            assert not any(
                (p1 == p2).any() for p1, p2 in zip(target_qvalue, target_qvalue2)
            )

        # check that policy is updated after parameter update
        actorp_set = set(actor.parameters())
        loss_fnp_set = set(loss_fn.parameters())
        assert len(actorp_set.intersection(loss_fnp_set)) == len(actorp_set)
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_td3bc_tensordict_keys(self, td_est):
        actor = self._create_mock_actor()
        value = self._create_mock_value()
        loss_fn = TD3BCLoss(
            actor,
            value,
            action_spec=actor.spec,
        )

        default_keys = {
            "priority": "td_error",
            "state_action_value": "state_action_value",
            "action": "action",
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
        }

        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
            td_est=td_est,
        )

        value = self._create_mock_value(out_keys=["state_action_value_test"])
        loss_fn = TD3BCLoss(
            actor,
            value,
            action_spec=actor.spec,
        )
        key_mapping = {
            "state_action_value": ("value", "state_action_value_test"),
            "reward": ("reward", "reward_test"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize("spec", [True, False])
    @pytest.mark.parametrize("bounds", [True, False])
    def test_constructor(self, spec, bounds):
        actor = self._create_mock_actor()
        value = self._create_mock_value()
        action_spec = actor.spec if spec else None
        bounds = (-1, 1) if bounds else None
        if (bounds is not None and action_spec is not None) or (
            bounds is None and action_spec is None
        ):
            with pytest.raises(ValueError, match="but not both"):
                TD3BCLoss(
                    actor,
                    value,
                    action_spec=action_spec,
                    bounds=bounds,
                )
            return
        TD3BCLoss(
            actor,
            value,
            action_spec=action_spec,
            bounds=bounds,
        )

    # TODO: test for action_key, atm the action key of the TD3+BC loss is not configurable,
    # since it is used in it's constructor
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    def test_td3bc_notensordict(
        self, observation_key, reward_key, done_key, terminated_key
    ):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(in_keys=[observation_key])
        qvalue = self._create_mock_value(
            observation_key=observation_key, out_keys=["state_action_value"]
        )
        td = self._create_mock_data_td3bc(
            observation_key=observation_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
        )
        loss = TD3BCLoss(actor, qvalue, action_spec=actor.spec)
        loss.set_keys(reward=reward_key, done=done_key, terminated=terminated_key)

        kwargs = {
            observation_key: td.get(observation_key),
            f"next_{reward_key}": td.get(("next", reward_key)),
            f"next_{done_key}": td.get(("next", done_key)),
            f"next_{terminated_key}": td.get(("next", terminated_key)),
            f"next_{observation_key}": td.get(("next", observation_key)),
            "action": td.get("action"),
        }
        td = TensorDict(kwargs, td.batch_size).unflatten_keys("_")

        with pytest.warns(
            UserWarning, match="No target network updater has been"
        ) if rl_warnings() else contextlib.nullcontext():
            torch.manual_seed(0)
            loss_val_td = loss(td)
            torch.manual_seed(0)
            loss_val = loss(**kwargs)
            loss_val_reconstruct = TensorDict(dict(zip(loss.out_keys, loss_val)), [])
            assert_allclose_td(loss_val_reconstruct, loss_val_td)

            # test select
            loss.select_out_keys("loss_actor", "loss_qvalue")
            torch.manual_seed(0)
            if torch.__version__ >= "2.0.0":
                loss_actor, loss_qvalue = loss(**kwargs)
            else:
                with pytest.raises(
                    RuntimeError,
                    match="You are likely using tensordict.nn.dispatch with keyword arguments",
                ):
                    loss_actor, loss_qvalue = loss(**kwargs)
                return

            assert loss_actor == loss_val_td["loss_actor"]
            assert loss_qvalue == loss_val_td["loss_qvalue"]

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_td3bc_reduction(self, reduction):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        td = self._create_mock_data_td3bc(device=device)
        action_spec = actor.spec
        bounds = None
        loss_fn = TD3BCLoss(
            actor,
            value,
            action_spec=action_spec,
            bounds=bounds,
            loss_function="l2",
            delay_qvalue=False,
            delay_actor=False,
            reduction=reduction,
        )
        loss_fn.make_value_estimator()
        loss = loss_fn(td)
        if reduction == "none":
            for key in loss.keys():
                if key.startswith("loss"):
                    assert loss[key].shape == td.shape
        else:
            for key in loss.keys():
                if not key.startswith("loss"):
                    continue
                assert loss[key].shape == torch.Size([])


@pytest.mark.skipif(
    not _has_functorch, reason=f"functorch not installed: {FUNCTORCH_ERR}"
)
@pytest.mark.parametrize("version", [1, 2])
class TestSAC(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        observation_key="observation",
        action_key="action",
        composite_action_dist=False,
        return_action_spec=False,
    ):
        # Actor
        action_spec = Bounded(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        if composite_action_dist:
            action_spec = Composite({action_key: {"action1": action_spec}})
        net = nn.Sequential(nn.Linear(obs_dim, 2 * action_dim), NormalParamExtractor())
        if composite_action_dist:
            distribution_class = functools.partial(
                CompositeDistribution,
                distribution_map={
                    "action1": TanhNormal,
                },
            )
            module_out_keys = [
                ("params", "action1", "loc"),
                ("params", "action1", "scale"),
            ]
            actor_in_keys = ["params"]
        else:
            distribution_class = TanhNormal
            module_out_keys = actor_in_keys = ["loc", "scale"]
        module = TensorDictModule(
            net, in_keys=[observation_key], out_keys=module_out_keys
        )
        actor = ProbabilisticActor(
            module=module,
            distribution_class=distribution_class,
            in_keys=actor_in_keys,
            out_keys=[action_key],
            spec=action_spec,
        )
        assert actor.log_prob_keys
        actor = actor.to(device)
        if return_action_spec:
            return actor, action_spec
        return actor

    def _create_mock_qvalue(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        observation_key="observation",
        action_key="action",
        out_keys=None,
    ):
        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(obs_dim + action_dim, 1)

            def forward(self, obs, act):
                if isinstance(act, TensorDictBase):
                    act = act.get("action1")
                return self.linear(torch.cat([obs, act], -1))

        module = ValueClass()
        qvalue = ValueOperator(
            module=module,
            in_keys=[observation_key, action_key],
            out_keys=out_keys,
        )
        return qvalue.to(device)

    def _create_mock_value(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        observation_key="observation",
        out_keys=None,
    ):
        module = nn.Linear(obs_dim, 1)
        value = ValueOperator(
            module=module,
            in_keys=[observation_key],
            out_keys=out_keys,
        )
        return value.to(device)

    def _create_mock_common_layer_setup(
        self,
        n_obs=3,
        n_act=4,
        ncells=4,
        batch=2,
        n_hidden=2,
        composite_action_dist=False,
    ):
        class QValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(n_hidden + n_act, n_hidden)
                self.relu = nn.ReLU()
                self.linear2 = nn.Linear(n_hidden, 1)

            def forward(self, obs, act):
                if isinstance(act, TensorDictBase):
                    act = act.get("action1")
                return self.linear2(self.relu(self.linear1(torch.cat([obs, act], -1))))

        common = MLP(
            num_cells=ncells,
            in_features=n_obs,
            depth=3,
            out_features=n_hidden,
        )
        actor_net = MLP(
            num_cells=ncells,
            in_features=n_hidden,
            depth=1,
            out_features=2 * n_act,
        )
        qvalue = QValueClass()
        batch = [batch]
        action = torch.randn(*batch, n_act)
        td = TensorDict(
            {
                "obs": torch.randn(*batch, n_obs),
                "action": {"action1": action} if composite_action_dist else action,
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
                "terminated": torch.zeros(*batch, 1, dtype=torch.bool),
                "next": {
                    "obs": torch.randn(*batch, n_obs),
                    "reward": torch.randn(*batch, 1),
                    "done": torch.zeros(*batch, 1, dtype=torch.bool),
                    "terminated": torch.zeros(*batch, 1, dtype=torch.bool),
                },
            },
            batch,
        )
        common = Mod(common, in_keys=["obs"], out_keys=["hidden"])
        if composite_action_dist:
            distribution_class = functools.partial(
                CompositeDistribution,
                distribution_map={
                    "action1": TanhNormal,
                },
            )
            module_out_keys = [
                ("params", "action1", "loc"),
                ("params", "action1", "scale"),
            ]
            actor_in_keys = ["params"]
        else:
            distribution_class = TanhNormal
            module_out_keys = actor_in_keys = ["loc", "scale"]
        actor = ProbSeq(
            common,
            Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
            Mod(NormalParamExtractor(), in_keys=["param"], out_keys=module_out_keys),
            ProbMod(
                in_keys=actor_in_keys,
                out_keys=["action"],
                distribution_class=distribution_class,
            ),
        )
        qvalue_head = Mod(
            qvalue, in_keys=["hidden", "action"], out_keys=["state_action_value"]
        )
        qvalue = Seq(common, qvalue_head)
        return actor, qvalue, common, td

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_sac(
        self,
        batch=16,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
        observation_key="observation",
        action_key="action",
        done_key="done",
        terminated_key="terminated",
        reward_key="reward",
        composite_action_dist=False,
    ):
        # create a tensordict
        obs = torch.randn(batch, obs_dim, device=device)
        next_obs = torch.randn(batch, obs_dim, device=device)
        if atoms:
            raise NotImplementedError
        else:
            action = torch.randn(batch, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, 1, device=device)
        done = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch,),
            source={
                observation_key: obs,
                "next": {
                    observation_key: next_obs,
                    done_key: done,
                    terminated_key: terminated,
                    reward_key: reward,
                },
                action_key: {"action1": action} if composite_action_dist else action,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_sac(
        self,
        batch=8,
        T=4,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
        composite_action_dist=False,
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim, device=device)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action = torch.randn(batch, T, atoms, action_dim, device=device).clamp(
                -1, 1
            )
        else:
            action = torch.randn(batch, T, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = torch.ones(batch, T, dtype=torch.bool, device=device)
        action = action.masked_fill_(~mask.unsqueeze(-1), 0.0)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "next": {
                    "observation": next_obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                    "done": done,
                    "terminated": terminated,
                    "reward": reward.masked_fill_(~mask.unsqueeze(-1), 0.0),
                },
                "collector": {"mask": mask},
                "action": {"action1": action} if composite_action_dist else action,
            },
            names=[None, "time"],
            device=device,
        )
        return td

    def test_reset_parameters_recursive(self, version):
        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()
        if version == 1:
            value = self._create_mock_value()
        else:
            value = None
        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
        )
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("delay_value", (True, False))
    @pytest.mark.parametrize("delay_actor", (True, False))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_sac(
        self,
        delay_value,
        delay_actor,
        delay_qvalue,
        num_qvalue,
        device,
        version,
        td_est,
        composite_action_dist,
    ):
        if (delay_actor or delay_qvalue) and not delay_value:
            pytest.skip("incompatible config")

        torch.manual_seed(self.seed)
        td = self._create_mock_data_sac(
            device=device, composite_action_dist=composite_action_dist
        )

        # For composite action distributions, we need to pass the action_spec
        # explicitly because ProbabilisticActor doesn't preserve it properly
        if composite_action_dist:
            actor, action_spec = self._create_mock_actor(
                device=device,
                composite_action_dist=composite_action_dist,
                return_action_spec=True,
            )
        else:
            actor = self._create_mock_actor(
                device=device, composite_action_dist=composite_action_dist
            )
            action_spec = None
        qvalue = self._create_mock_qvalue(device=device)
        if version == 1:
            value = self._create_mock_value(device=device)
        else:
            value = None

        kwargs = {}
        if delay_actor:
            kwargs["delay_actor"] = True
        if delay_qvalue:
            kwargs["delay_qvalue"] = True
        if delay_value:
            kwargs["delay_value"] = True

        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            action_spec=action_spec,
            **kwargs,
        )

        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)

        with _check_td_steady(td), pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
            loss = loss_fn(td)

        assert loss_fn.tensor_keys.priority in td.keys()

        # check that losses are independent
        for k in loss.keys():
            if not k.startswith("loss"):
                continue
            loss[k].sum().backward(retain_graph=True)
            if k == "loss_actor":
                if version == 1:
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.value_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            elif k == "loss_value" and version == 1:
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            elif k == "loss_qvalue":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                if version == 1:
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.value_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            elif k == "loss_alpha":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                if version == 1:
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.value_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()

        sum(
            [item for name, item in loss.items() if name.startswith("loss_")]
        ).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            if not name.startswith("target_"):
                assert (
                    p.grad is not None and p.grad.norm() > 0.0
                ), f"parameter {name} (shape: {p.shape}) has a null gradient"
            else:
                assert (
                    p.grad is None or p.grad.norm() == 0.0
                ), f"target parameter {name} (shape: {p.shape}) has a non-null gradient"

    @pytest.mark.parametrize("delay_value", (True,))
    @pytest.mark.parametrize("delay_actor", (True,))
    @pytest.mark.parametrize("delay_qvalue", (True,))
    @pytest.mark.parametrize("num_qvalue", [2])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("td_est", [None])
    @pytest.mark.parametrize("composite_action_dist", [False])
    def test_sac_deactivate_vmap(
        self,
        delay_value,
        delay_actor,
        delay_qvalue,
        num_qvalue,
        device,
        version,
        td_est,
        composite_action_dist,
    ):
        if (delay_actor or delay_qvalue) and not delay_value:
            pytest.skip("incompatible config")

        torch.manual_seed(self.seed)
        td = self._create_mock_data_sac(
            device=device, composite_action_dist=composite_action_dist
        )

        actor = self._create_mock_actor(
            device=device, composite_action_dist=composite_action_dist
        )
        qvalue = self._create_mock_qvalue(device=device)
        if version == 1:
            value = self._create_mock_value(device=device)
        else:
            value = None

        kwargs = {}
        if delay_actor:
            kwargs["delay_actor"] = True
        if delay_qvalue:
            kwargs["delay_qvalue"] = True
        if delay_value:
            kwargs["delay_value"] = True

        torch.manual_seed(0)
        loss_fn_vmap = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            deactivate_vmap=False,
            **kwargs,
        )

        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn_vmap.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn_vmap.make_value_estimator(td_est)

        tdc = td.clone()
        torch.manual_seed(0)
        with _check_td_steady(td), pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
            loss_vmap = loss_fn_vmap(td)
        td = tdc
        torch.manual_seed(0)
        loss_fn_no_vmap = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            deactivate_vmap=True,
            **kwargs,
        )

        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn_no_vmap.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn_no_vmap.make_value_estimator(td_est)

        torch.manual_seed(0)
        with pytest.raises(
            NotImplementedError,
            match="This implementation is not supported for torch<2.7",
        ) if torch.__version__ < "2.7" else contextlib.nullcontext():
            with _check_td_steady(td), pytest.warns(
                UserWarning, match="No target network updater"
            ) if rl_warnings() else contextlib.nullcontext():
                loss_no_vmap = loss_fn_no_vmap(td)
            assert_allclose_td(loss_vmap, loss_no_vmap)

    @pytest.mark.parametrize("delay_value", (True, False))
    @pytest.mark.parametrize("delay_actor", (True, False))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [2])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_sac_state_dict(
        self,
        delay_value,
        delay_actor,
        delay_qvalue,
        num_qvalue,
        device,
        version,
        composite_action_dist,
    ):
        if (delay_actor or delay_qvalue) and not delay_value:
            pytest.skip("incompatible config")

        torch.manual_seed(self.seed)

        # For composite action distributions, we need to pass the action_spec
        # explicitly because ProbabilisticActor doesn't preserve it properly
        if composite_action_dist:
            actor, action_spec = self._create_mock_actor(
                device=device,
                composite_action_dist=composite_action_dist,
                return_action_spec=True,
            )
        else:
            actor = self._create_mock_actor(
                device=device, composite_action_dist=composite_action_dist
            )
            action_spec = None
        qvalue = self._create_mock_qvalue(device=device)
        if version == 1:
            value = self._create_mock_value(device=device)
        else:
            value = None

        kwargs = {}
        if delay_actor:
            kwargs["delay_actor"] = True
        if delay_qvalue:
            kwargs["delay_qvalue"] = True
        if delay_value:
            kwargs["delay_value"] = True

        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            action_spec=action_spec,
            **kwargs,
        )
        sd = loss_fn.state_dict()
        loss_fn2 = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            action_spec=action_spec,
            **kwargs,
        )
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("separate_losses", [False, True])
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_sac_separate_losses(
        self,
        device,
        separate_losses,
        version,
        composite_action_dist,
        n_act=4,
    ):
        torch.manual_seed(self.seed)
        actor, qvalue, common, td = self._create_mock_common_layer_setup(
            n_act=n_act, composite_action_dist=composite_action_dist
        )

        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            action_spec=Unbounded(shape=(n_act,)),
            num_qvalue_nets=1,
            separate_losses=separate_losses,
        )
        with pytest.warns(
            UserWarning, match="No target network updater has been"
        ) if rl_warnings() else contextlib.nullcontext():
            loss = loss_fn(td)

            assert loss_fn.tensor_keys.priority in td.keys()

            # check that losses are independent
            for k in loss.keys():
                if not k.startswith("loss"):
                    continue
                loss[k].sum().backward(retain_graph=True)
                if k == "loss_actor":
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.qvalue_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                elif k == "loss_qvalue":
                    common_layers_no = len(list(common.parameters()))
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                    if separate_losses:
                        common_layers = itertools.islice(
                            loss_fn.qvalue_network_params.values(True, True),
                            common_layers_no,
                        )
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in common_layers
                        )
                        qvalue_layers = itertools.islice(
                            loss_fn.qvalue_network_params.values(True, True),
                            common_layers_no,
                            None,
                        )
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in qvalue_layers
                        )
                    else:
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in loss_fn.qvalue_network_params.values(True, True)
                        )
                elif k == "loss_alpha":
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.qvalue_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                else:
                    raise NotImplementedError(k)
                loss_fn.zero_grad()

    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("delay_value", (True, False))
    @pytest.mark.parametrize("delay_actor", (True, False))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_sac_batcher(
        self,
        n,
        delay_value,
        delay_actor,
        delay_qvalue,
        num_qvalue,
        device,
        version,
        composite_action_dist,
    ):
        if (delay_actor or delay_qvalue) and not delay_value:
            pytest.skip("incompatible config")
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_sac(
            device=device, composite_action_dist=composite_action_dist
        )

        # For composite action distributions, we need to pass the action_spec
        # explicitly because ProbabilisticActor doesn't preserve it properly
        if composite_action_dist:
            actor, action_spec = self._create_mock_actor(
                device=device,
                composite_action_dist=composite_action_dist,
                return_action_spec=True,
            )
        else:
            actor = self._create_mock_actor(
                device=device, composite_action_dist=composite_action_dist
            )
            action_spec = None
        qvalue = self._create_mock_qvalue(device=device)
        if version == 1:
            value = self._create_mock_value(device=device)
        else:
            value = None

        kwargs = {}
        if delay_actor:
            kwargs["delay_actor"] = True
        if delay_qvalue:
            kwargs["delay_qvalue"] = True
        if delay_value:
            kwargs["delay_value"] = True

        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            action_spec=action_spec,
            **kwargs,
        )

        ms = MultiStep(gamma=0.9, n_steps=n).to(device)

        td_clone = td.clone()
        ms_td = ms(td_clone)

        torch.manual_seed(0)
        np.random.seed(0)
        with pytest.warns(
            UserWarning,
            match="No target network updater has been associated with this loss module",
        ) if rl_warnings() else contextlib.nullcontext():
            with _check_td_steady(ms_td):
                loss_ms = loss_fn(ms_td)
            assert loss_fn.tensor_keys.priority in ms_td.keys()

            with torch.no_grad():
                torch.manual_seed(0)  # log-prob is computed with a random action
                np.random.seed(0)
                loss = loss_fn(td)
            if n == 1:
                assert_allclose_td(td, ms_td.select(*list(td.keys(True, True))))
                _loss = sum(
                    [item for name, item in loss.items() if name.startswith("loss_")]
                )
                _loss_ms = sum(
                    [item for name, item in loss_ms.items() if name.startswith("loss_")]
                )
                assert (
                    abs(_loss - _loss_ms) < 1e-3
                ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
            else:
                with pytest.raises(AssertionError):
                    assert_allclose_td(loss, loss_ms)
            sum(
                [item for name, item in loss_ms.items() if name.startswith("loss_")]
            ).backward()
            named_parameters = loss_fn.named_parameters()
            for name, p in named_parameters:
                if not name.startswith("target_"):
                    assert (
                        p.grad is not None and p.grad.norm() > 0.0
                    ), f"parameter {name} (shape: {p.shape}) has a null gradient"
                else:
                    assert (
                        p.grad is None or p.grad.norm() == 0.0
                    ), f"target parameter {name} (shape: {p.shape}) has a non-null gradient"

            # Check param update effect on targets
            target_actor = [
                p.clone()
                for p in loss_fn.target_actor_network_params.values(
                    include_nested=True, leaves_only=True
                )
            ]
            target_qvalue = [
                p.clone()
                for p in loss_fn.target_qvalue_network_params.values(
                    include_nested=True, leaves_only=True
                )
            ]
            if version == 1:
                target_value = [
                    p.clone()
                    for p in loss_fn.target_value_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                ]
            for p in loss_fn.parameters():
                if p.requires_grad:
                    p.data += torch.randn_like(p)
            target_actor2 = [
                p.clone()
                for p in loss_fn.target_actor_network_params.values(
                    include_nested=True, leaves_only=True
                )
            ]
            target_qvalue2 = [
                p.clone()
                for p in loss_fn.target_qvalue_network_params.values(
                    include_nested=True, leaves_only=True
                )
            ]
            if version == 1:
                target_value2 = [
                    p.clone()
                    for p in loss_fn.target_value_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                ]
            if loss_fn.delay_actor:
                assert all(
                    (p1 == p2).all() for p1, p2 in zip(target_actor, target_actor2)
                )
            else:
                assert not any(
                    (p1 == p2).any() for p1, p2 in zip(target_actor, target_actor2)
                )
            if loss_fn.delay_qvalue:
                assert all(
                    (p1 == p2).all() for p1, p2 in zip(target_qvalue, target_qvalue2)
                )
            else:
                assert not any(
                    (p1 == p2).any() for p1, p2 in zip(target_qvalue, target_qvalue2)
                )
            if version == 1:
                if loss_fn.delay_value:
                    assert all(
                        (p1 == p2).all() for p1, p2 in zip(target_value, target_value2)
                    )
                else:
                    assert not any(
                        (p1 == p2).any() for p1, p2 in zip(target_value, target_value2)
                    )

            # check that policy is updated after parameter update
            parameters = [p.clone() for p in actor.parameters()]
            for p in loss_fn.parameters():
                if p.requires_grad:
                    p.data += torch.randn_like(p)
            assert all(
                (p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters())
            )

    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_sac_tensordict_keys(self, td_est, version, composite_action_dist):
        td = self._create_mock_data_sac(composite_action_dist=composite_action_dist)

        # For composite action distributions, we need to pass the action_spec
        # explicitly because ProbabilisticActor doesn't preserve it properly
        if composite_action_dist:
            actor, action_spec = self._create_mock_actor(
                composite_action_dist=composite_action_dist,
                return_action_spec=True,
            )
        else:
            actor = self._create_mock_actor(composite_action_dist=composite_action_dist)
            action_spec = None
        qvalue = self._create_mock_qvalue()
        if version == 1:
            value = self._create_mock_value()
        else:
            value = None

        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=2,
            loss_function="l2",
            action_spec=action_spec,
        )

        default_keys = {
            "priority": "td_error",
            "value": "state_value",
            "state_action_value": "state_action_value",
            "action": "action",
            "log_prob": "action_log_prob",
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
        }

        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
            td_est=td_est,
        )

        value = self._create_mock_value()
        loss_fn = SACLoss(
            actor,
            value,
            loss_function="l2",
        )

        key_mapping = {
            "value": ("value", "state_value_test"),
            "reward": ("reward", "reward_test"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize("action_key", ["action", "action2"])
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    def test_sac_notensordict(
        self, action_key, observation_key, reward_key, done_key, terminated_key, version
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_sac(
            action_key=action_key,
            observation_key=observation_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
        )

        actor = self._create_mock_actor(
            observation_key=observation_key, action_key=action_key
        )
        qvalue = self._create_mock_qvalue(
            observation_key=observation_key,
            action_key=action_key,
            out_keys=["state_action_value"],
        )
        if version == 1:
            value = self._create_mock_value(observation_key=observation_key)
        else:
            value = None

        loss = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
        )
        loss.set_keys(
            action=action_key,
            reward=reward_key,
            done=done_key,
            terminated=terminated_key,
        )

        kwargs = {
            action_key: td.get(action_key),
            observation_key: td.get(observation_key),
            f"next_{reward_key}": td.get(("next", reward_key)),
            f"next_{done_key}": td.get(("next", done_key)),
            f"next_{terminated_key}": td.get(("next", terminated_key)),
            f"next_{observation_key}": td.get(("next", observation_key)),
        }
        td = TensorDict(kwargs, td.batch_size).unflatten_keys("_")

        # setting the seed for each loss so that drawing the random samples from value network
        # leads to same numbers for both runs
        torch.manual_seed(self.seed)
        with pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
            loss_val = loss(**kwargs)

        torch.manual_seed(self.seed)

        SoftUpdate(loss, eps=0.5)

        loss_val_td = loss(td)

        if version == 1:
            assert len(loss_val) == 6
        elif version == 2:
            assert len(loss_val) == 5

        torch.testing.assert_close(loss_val_td.get("loss_actor"), loss_val[0])
        torch.testing.assert_close(loss_val_td.get("loss_qvalue"), loss_val[1])
        torch.testing.assert_close(loss_val_td.get("loss_alpha"), loss_val[2])
        torch.testing.assert_close(loss_val_td.get("alpha"), loss_val[3])
        torch.testing.assert_close(loss_val_td.get("entropy"), loss_val[4])
        if version == 1:
            torch.testing.assert_close(loss_val_td.get("loss_value"), loss_val[5])
        # test select
        torch.manual_seed(self.seed)
        loss.select_out_keys("loss_actor", "loss_alpha")
        if torch.__version__ >= "2.0.0":
            loss_actor, loss_alpha = loss(**kwargs)
        else:
            with pytest.raises(
                RuntimeError,
                match="You are likely using tensordict.nn.dispatch with keyword arguments",
            ):
                loss_actor, loss_alpha = loss(**kwargs)
            return
        assert loss_actor == loss_val_td["loss_actor"]
        assert loss_alpha == loss_val_td["loss_alpha"]

    @pytest.mark.parametrize("action_key", ["action", "action2"])
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    def test_sac_terminating(
        self, action_key, observation_key, reward_key, done_key, terminated_key, version
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_sac(
            action_key=action_key,
            observation_key=observation_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
        )

        actor = self._create_mock_actor(
            observation_key=observation_key, action_key=action_key
        )
        qvalue = self._create_mock_qvalue(
            observation_key=observation_key,
            action_key=action_key,
            out_keys=["state_action_value"],
        )
        if version == 1:
            value = self._create_mock_value(observation_key=observation_key)
        else:
            value = None

        loss = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            skip_done_states=True,
        )
        loss.set_keys(
            action=action_key,
            reward=reward_key,
            done=done_key,
            terminated=terminated_key,
        )

        torch.manual_seed(self.seed)

        SoftUpdate(loss, eps=0.5)

        done = td.get(("next", done_key))
        while not (done.any() and not done.all()):
            done.bernoulli_(0.1)
        obs_nan = td.get(("next", terminated_key))
        obs_nan[done.squeeze(-1)] = float("nan")

        kwargs = {
            action_key: td.get(action_key),
            observation_key: td.get(observation_key),
            f"next_{reward_key}": td.get(("next", reward_key)),
            f"next_{done_key}": done,
            f"next_{terminated_key}": obs_nan,
            f"next_{observation_key}": td.get(("next", observation_key)),
        }
        td = TensorDict(kwargs, td.batch_size).unflatten_keys("_")
        assert loss(td).isfinite().all()

    def test_state_dict(self, version):
        if version == 1:
            pytest.skip("Test not implemented for version 1.")
        model = torch.nn.Linear(3, 4)
        actor_module = TensorDictModule(model, in_keys=["obs"], out_keys=["logits"])
        policy = ProbabilisticActor(
            module=actor_module,
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=TanhDelta,
        )
        value = ValueOperator(module=model, in_keys=["obs"], out_keys="value")

        loss = SACLoss(
            actor_network=policy,
            qvalue_network=value,
            action_spec=Unbounded(shape=(2,)),
        )
        state = loss.state_dict()

        loss = SACLoss(
            actor_network=policy,
            qvalue_network=value,
            action_spec=Unbounded(shape=(2,)),
        )
        loss.load_state_dict(state)

        # with an access in between
        loss = SACLoss(
            actor_network=policy,
            qvalue_network=value,
            action_spec=Unbounded(shape=(2,)),
        )
        loss.target_entropy
        state = loss.state_dict()

        loss = SACLoss(
            actor_network=policy,
            qvalue_network=value,
            action_spec=Unbounded(shape=(2,)),
        )
        loss.load_state_dict(state)

    @pytest.mark.parametrize("action_dim", [1, 2, 4, 8])
    def test_sac_target_entropy_auto(self, version, action_dim):
        """Regression test for issue #3291: target_entropy='auto' should be -dim(A)."""
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(action_dim=action_dim)
        qvalue = self._create_mock_qvalue(action_dim=action_dim)
        if version == 1:
            value = self._create_mock_value(action_dim=action_dim)
        else:
            value = None

        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
        )
        # target_entropy="auto" should compute -action_dim
        assert (
            loss_fn.target_entropy.item() == -action_dim
        ), f"target_entropy should be -{action_dim}, got {loss_fn.target_entropy.item()}"

    @pytest.mark.parametrize("target_entropy", [-1.0, -2.0, -5.0, 0.0])
    def test_sac_target_entropy_explicit(self, version, target_entropy):
        """Regression test for explicit target_entropy values."""
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()
        if version == 1:
            value = self._create_mock_value()
        else:
            value = None

        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            target_entropy=target_entropy,
        )
        assert (
            loss_fn.target_entropy.item() == target_entropy
        ), f"target_entropy should be {target_entropy}, got {loss_fn.target_entropy.item()}"

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_sac_reduction(self, reduction, version, composite_action_dist):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        td = self._create_mock_data_sac(
            device=device, composite_action_dist=composite_action_dist
        )
        # For composite action distributions, we need to pass the action_spec
        # explicitly because ProbabilisticActor doesn't preserve it properly
        if composite_action_dist:
            actor, action_spec = self._create_mock_actor(
                device=device,
                composite_action_dist=composite_action_dist,
                return_action_spec=True,
            )
        else:
            actor = self._create_mock_actor(
                device=device, composite_action_dist=composite_action_dist
            )
            action_spec = None
        qvalue = self._create_mock_qvalue(device=device)
        if version == 1:
            value = self._create_mock_value(device=device)
        else:
            value = None
        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            loss_function="l2",
            delay_qvalue=False,
            delay_actor=False,
            delay_value=False,
            reduction=reduction,
            action_spec=action_spec,
            scalar_output_mode="exclude" if reduction == "none" else None,
        )
        loss_fn.make_value_estimator()
        loss = loss_fn(td)
        if reduction == "none":
            for key in loss.keys():
                if key.startswith("loss"):
                    assert loss[key].shape == td.shape
        else:
            for key in loss.keys():
                if not key.startswith("loss"):
                    continue
                assert loss[key].shape == torch.Size([])

    def test_sac_prioritized_weights(self, version):
        """Test SAC with prioritized replay buffer weighted loss reduction."""
        if version != 2:
            pytest.skip("Test not implemented for version 1.")
        torch.manual_seed(42)
        n_obs = 4
        n_act = 2
        batch_size = 32
        buffer_size = 100

        # Actor network
        actor_net = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * n_act),
            NormalParamExtractor(),
        )
        actor_module = TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        actor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            return_log_prob=True,
            spec=Bounded(
                low=-torch.ones(n_act), high=torch.ones(n_act), shape=(n_act,)
            ),
        )

        # Q-value network
        qvalue_net = MLP(in_features=n_obs + n_act, out_features=1, num_cells=[64, 64])
        qvalue = ValueOperator(module=qvalue_net, in_keys=["observation", "action"])

        # Value network for SAC v1
        value_net = MLP(in_features=n_obs, out_features=1, num_cells=[64, 64])
        value = ValueOperator(module=value_net, in_keys=["observation"])

        # Create SAC loss
        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=2,
        )
        SoftUpdate(loss_fn, eps=0.5)
        loss_fn.make_value_estimator()

        # Create prioritized replay buffer
        rb = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.9,
            storage=LazyTensorStorage(buffer_size),
            batch_size=batch_size,
            priority_key="td_error",
        )

        # Create initial data
        initial_data = TensorDict(
            {
                "observation": torch.randn(buffer_size, n_obs),
                "action": torch.randn(buffer_size, n_act).clamp(-1, 1),
                ("next", "observation"): torch.randn(buffer_size, n_obs),
                ("next", "reward"): torch.randn(buffer_size, 1),
                ("next", "done"): torch.zeros(buffer_size, 1, dtype=torch.bool),
                ("next", "terminated"): torch.zeros(buffer_size, 1, dtype=torch.bool),
            },
            batch_size=[buffer_size],
        )
        rb.extend(initial_data)

        # Sample (weights should all be identical initially)
        sample1 = rb.sample()
        assert "priority_weight" in sample1.keys()
        weights1 = sample1["priority_weight"]
        assert torch.allclose(weights1, weights1[0], atol=1e-5)

        # Run loss to get priorities
        loss_fn(sample1)
        assert "td_error" in sample1.keys()

        # Update replay buffer with new priorities
        rb.update_tensordict_priority(sample1)

        # Sample again - weights should now be non-equal
        sample2 = rb.sample()
        weights2 = sample2["priority_weight"]
        assert weights2.std() > 1e-5

        # Run loss again with varied weights
        loss_out2 = loss_fn(sample2)
        assert torch.isfinite(loss_out2["loss_qvalue"])

        # Verify weighted vs unweighted differ
        loss_fn_no_weights = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=2,
            use_prioritized_weights=False,
        )
        SoftUpdate(loss_fn_no_weights, eps=0.5)
        loss_fn_no_weights.make_value_estimator()
        loss_fn_no_weights.qvalue_network_params = loss_fn.qvalue_network_params
        loss_fn_no_weights.target_qvalue_network_params = (
            loss_fn.target_qvalue_network_params
        )
        loss_fn_no_weights.actor_network_params = loss_fn.actor_network_params
        loss_fn_no_weights.value_network_params = loss_fn.value_network_params
        loss_fn_no_weights.target_value_network_params = (
            loss_fn.target_value_network_params
        )

        loss_out_no_weights = loss_fn_no_weights(sample2)
        # Weighted and unweighted should differ (in general)
        assert torch.isfinite(loss_out_no_weights["loss_qvalue"])


@pytest.mark.skipif(
    not _has_functorch, reason=f"functorch not installed: {FUNCTORCH_ERR}"
)
class TestDiscreteSAC(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        observation_key="observation",
        action_key="action",
    ):
        # Actor
        action_spec = OneHot(action_dim)
        net = nn.Linear(obs_dim, action_dim)
        module = TensorDictModule(net, in_keys=[observation_key], out_keys=["logits"])
        actor = ProbabilisticActor(
            spec=action_spec,
            module=module,
            in_keys=["logits"],
            out_keys=[action_key],
            distribution_class=OneHotCategorical,
            return_log_prob=False,
        )
        return actor.to(device)

    def _create_mock_qvalue(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        observation_key="observation",
    ):
        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(obs_dim, action_dim)

            def forward(self, obs):
                return self.linear(obs)

        module = ValueClass()
        qvalue = ValueOperator(
            module=module, in_keys=[observation_key], out_keys=["action_value"]
        )
        return qvalue.to(device)

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_sac(
        self,
        batch=16,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
        observation_key="observation",
        action_key="action",
        done_key="done",
        terminated_key="terminated",
        reward_key="reward",
    ):
        # create a tensordict
        obs = torch.randn(batch, obs_dim, device=device)
        next_obs = torch.randn(batch, obs_dim, device=device)
        if atoms:
            action_value = torch.randn(batch, atoms, action_dim).softmax(-2)
            action = (
                (action_value[..., 0, :] == action_value[..., 0, :].max(-1, True)[0])
                .to(torch.long)
                .to(device)
            )
        else:
            action_value = torch.randn(batch, action_dim, device=device)
            action = (action_value == action_value.max(-1, True)[0]).to(torch.long)
        reward = torch.randn(batch, 1, device=device)
        done = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch,),
            source={
                observation_key: obs,
                "next": {
                    observation_key: next_obs,
                    done_key: done,
                    terminated_key: terminated,
                    reward_key: reward,
                },
                action_key: action,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_sac(
        self, batch=8, T=4, obs_dim=3, action_dim=4, atoms=None, device="cpu"
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim, device=device)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action_value = torch.randn(
                batch, T, atoms, action_dim, device=device
            ).softmax(-2)
            action = (
                action_value[..., 0, :] == action_value[..., 0, :].max(-1, True)[0]
            ).to(torch.long)
        else:
            action_value = torch.randn(batch, T, action_dim, device=device)
            action = (action_value == action_value.max(-1, True)[0]).to(torch.long)

        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = ~torch.zeros(batch, T, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "next": {
                    "observation": next_obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                    "done": done,
                    "terminated": terminated,
                    "reward": reward.masked_fill_(~mask.unsqueeze(-1), 0.0),
                },
                "collector": {"mask": mask},
                "action": action.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "action_value": action_value.masked_fill_(~mask.unsqueeze(-1), 0.0),
            },
            names=[None, "time"],
        )
        return td

    def test_reset_parameters_recursive(self):
        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()
        loss_fn = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=actor.spec["action"].space.n,
            action_space="one-hot",
        )
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [2])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("target_entropy_weight", [0.01, 0.5, 0.99])
    @pytest.mark.parametrize("target_entropy", ["auto", 1.0, 0.1, 0.0])
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_discrete_sac(
        self,
        delay_qvalue,
        num_qvalue,
        device,
        target_entropy_weight,
        target_entropy,
        td_est,
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_sac(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        kwargs = {}
        if delay_qvalue:
            kwargs["delay_qvalue"] = True

        loss_fn = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=actor.spec["action"].space.n,
            num_qvalue_nets=num_qvalue,
            target_entropy_weight=target_entropy_weight,
            target_entropy=target_entropy,
            loss_function="l2",
            action_space="one-hot",
            **kwargs,
        )
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)

        with _check_td_steady(td), pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
            loss = loss_fn(td)

        assert loss_fn.tensor_keys.priority in td.keys()

        # check that losses are independent
        for k in loss.keys():
            if not k.startswith("loss"):
                continue
            loss[k].sum().backward(retain_graph=True)
            if k == "loss_actor":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            elif k == "loss_qvalue":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            elif k == "loss_alpha":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()

        sum(
            [item for name, item in loss.items() if name.startswith("loss_")]
        ).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            if not name.startswith("target_"):
                assert (
                    p.grad is not None and p.grad.norm() > 0.0
                ), f"parameter {name} (shape: {p.shape}) has a null gradient"
            else:
                assert (
                    p.grad is None or p.grad.norm() == 0.0
                ), f"target parameter {name} (shape: {p.shape}) has a non-null gradient"

    @pytest.mark.parametrize("delay_qvalue", (True,))
    @pytest.mark.parametrize("num_qvalue", [2])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("target_entropy_weight", [0.5])
    @pytest.mark.parametrize("target_entropy", ["auto"])
    @pytest.mark.parametrize("td_est", [None])
    def test_discrete_sac_deactivate_vmap(
        self,
        delay_qvalue,
        num_qvalue,
        device,
        target_entropy_weight,
        target_entropy,
        td_est,
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_sac(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        kwargs = {}
        if delay_qvalue:
            kwargs["delay_qvalue"] = True

        torch.manual_seed(0)
        loss_fn_vmap = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=actor.spec["action"].space.n,
            num_qvalue_nets=num_qvalue,
            target_entropy_weight=target_entropy_weight,
            target_entropy=target_entropy,
            loss_function="l2",
            action_space="one-hot",
            deactivate_vmap=False,
            **kwargs,
        )
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn_vmap.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn_vmap.make_value_estimator(td_est)

        tdc = td.clone()
        with _check_td_steady(td), pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
            torch.manual_seed(1)
            loss_vmap = loss_fn_vmap(td)
        td = tdc

        torch.manual_seed(0)
        loss_fn_no_vmap = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=actor.spec["action"].space.n,
            num_qvalue_nets=num_qvalue,
            target_entropy_weight=target_entropy_weight,
            target_entropy=target_entropy,
            loss_function="l2",
            action_space="one-hot",
            deactivate_vmap=True,
            **kwargs,
        )
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn_no_vmap.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn_no_vmap.make_value_estimator(td_est)

        with pytest.raises(
            NotImplementedError,
            match="This implementation is not supported for torch<2.7",
        ) if torch.__version__ < "2.7" else contextlib.nullcontext():
            with _check_td_steady(td), pytest.warns(
                UserWarning, match="No target network updater"
            ) if rl_warnings() else contextlib.nullcontext():
                torch.manual_seed(1)
                loss_no_vmap = loss_fn_no_vmap(td)
            assert_allclose_td(loss_vmap, loss_no_vmap)

    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [2])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("target_entropy_weight", [0.5])
    @pytest.mark.parametrize("target_entropy", ["auto"])
    def test_discrete_sac_state_dict(
        self,
        delay_qvalue,
        num_qvalue,
        device,
        target_entropy_weight,
        target_entropy,
    ):
        torch.manual_seed(self.seed)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        kwargs = {}
        if delay_qvalue:
            kwargs["delay_qvalue"] = True

        loss_fn = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=actor.spec["action"].space.n,
            num_qvalue_nets=num_qvalue,
            target_entropy_weight=target_entropy_weight,
            target_entropy=target_entropy,
            loss_function="l2",
            action_space="one-hot",
            **kwargs,
        )
        sd = loss_fn.state_dict()
        loss_fn2 = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=actor.spec["action"].space.n,
            num_qvalue_nets=num_qvalue,
            target_entropy_weight=target_entropy_weight,
            target_entropy=target_entropy,
            loss_function="l2",
            action_space="one-hot",
            **kwargs,
        )
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("action_dim", [2, 4, 8])
    @pytest.mark.parametrize("target_entropy_weight", [0.5, 0.98])
    def test_discrete_sac_target_entropy_auto(self, action_dim, target_entropy_weight):
        """Regression test for target_entropy='auto' in DiscreteSACLoss."""
        import numpy as np

        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(action_dim=action_dim)
        qvalue = self._create_mock_qvalue(action_dim=action_dim)

        loss_fn = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=action_dim,
            target_entropy_weight=target_entropy_weight,
            action_space="one-hot",
        )
        # target_entropy="auto" should compute -log(1/num_actions) * target_entropy_weight
        expected = -float(np.log(1.0 / action_dim) * target_entropy_weight)
        assert (
            abs(loss_fn.target_entropy.item() - expected) < 1e-5
        ), f"target_entropy should be {expected}, got {loss_fn.target_entropy.item()}"

    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [2])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("target_entropy_weight", [0.01, 0.5, 0.99])
    @pytest.mark.parametrize("target_entropy", ["auto", 1.0, 0.1, 0.0])
    def test_discrete_sac_batcher(
        self,
        n,
        delay_qvalue,
        num_qvalue,
        device,
        target_entropy_weight,
        target_entropy,
        gamma=0.9,
    ):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_sac(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        kwargs = {}
        if delay_qvalue:
            kwargs["delay_qvalue"] = True
        loss_fn = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=actor.spec["action"].space.n,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            target_entropy_weight=target_entropy_weight,
            target_entropy=target_entropy,
            action_space="one-hot",
            **kwargs,
        )

        ms = MultiStep(gamma=gamma, n_steps=n).to(device)

        td_clone = td.clone()
        ms_td = ms(td_clone)

        torch.manual_seed(0)
        np.random.seed(0)
        with _check_td_steady(ms_td), pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
            loss_ms = loss_fn(ms_td)
        assert loss_fn.tensor_keys.priority in ms_td.keys()

        SoftUpdate(loss_fn, eps=0.5)

        with torch.no_grad():
            torch.manual_seed(0)  # log-prob is computed with a random action
            np.random.seed(0)
            loss = loss_fn(td)
        if n == 1:
            assert_allclose_td(td, ms_td.select(*list(td.keys(True, True))))
            _loss = sum(
                [item for name, item in loss.items() if name.startswith("loss_")]
            )
            _loss_ms = sum(
                [item for name, item in loss_ms.items() if name.startswith("loss_")]
            )
            assert (
                abs(_loss - _loss_ms) < 1e-3
            ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                assert_allclose_td(loss, loss_ms)
        sum(
            [item for name, item in loss_ms.items() if name.startswith("loss_")]
        ).backward()
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if not name.startswith("target_"):
                assert (
                    p.grad is not None and p.grad.norm() > 0.0
                ), f"parameter {name} (shape: {p.shape}) has a null gradient"
            else:
                assert (
                    p.grad is None or p.grad.norm() == 0.0
                ), f"target parameter {name} (shape: {p.shape}) has a non-null gradient"

        # Check param update effect on targets
        target_actor = [
            p.clone()
            for p in loss_fn.target_actor_network_params.values(
                include_nested=True, leaves_only=True
            )
        ]
        target_qvalue = [
            p.clone()
            for p in loss_fn.target_qvalue_network_params.values(
                include_nested=True, leaves_only=True
            )
        ]
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        target_actor2 = [
            p.clone()
            for p in loss_fn.target_actor_network_params.values(
                include_nested=True, leaves_only=True
            )
        ]
        target_qvalue2 = [
            p.clone()
            for p in loss_fn.target_qvalue_network_params.values(
                include_nested=True, leaves_only=True
            )
        ]
        if loss_fn.delay_actor:
            assert all((p1 == p2).all() for p1, p2 in zip(target_actor, target_actor2))
        else:
            assert not any(
                (p1 == p2).any() for p1, p2 in zip(target_actor, target_actor2)
            )
        if loss_fn.delay_qvalue:
            assert all(
                (p1 == p2).all() for p1, p2 in zip(target_qvalue, target_qvalue2)
            )
        else:
            assert not any(
                (p1 == p2).any() for p1, p2 in zip(target_qvalue, target_qvalue2)
            )

        # check that policy is updated after parameter update
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_discrete_sac_tensordict_keys(self, td_est):
        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()

        loss_fn = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=actor.spec["action"].space.n,
            loss_function="l2",
            action_space="one-hot",
        )

        default_keys = {
            "priority": "td_error",
            "value": "state_value",
            "action": "action",
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
        }
        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
            td_est=td_est,
        )

        qvalue = self._create_mock_qvalue()
        loss_fn = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=actor.spec["action"].space.n,
            loss_function="l2",
            action_space="one-hot",
        )

        key_mapping = {
            "value": ("value", "state_value_test"),
            "reward": ("reward", "reward_test"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize("action_key", ["action", "action2"])
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    def test_discrete_sac_notensordict(
        self, action_key, observation_key, reward_key, done_key, terminated_key
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_sac(
            action_key=action_key,
            observation_key=observation_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
        )

        actor = self._create_mock_actor(
            observation_key=observation_key, action_key=action_key
        )
        qvalue = self._create_mock_qvalue(
            observation_key=observation_key,
        )

        loss = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=actor.spec[action_key].space.n,
            action_space="one-hot",
        )
        loss.set_keys(
            action=action_key,
            reward=reward_key,
            done=done_key,
            terminated=terminated_key,
        )

        kwargs = {
            action_key: td.get(action_key),
            observation_key: td.get(observation_key),
            f"next_{reward_key}": td.get(("next", reward_key)),
            f"next_{done_key}": td.get(("next", done_key)),
            f"next_{terminated_key}": td.get(("next", terminated_key)),
            f"next_{observation_key}": td.get(("next", observation_key)),
        }
        td = TensorDict(kwargs, td.batch_size).unflatten_keys("_")

        with pytest.warns(
            UserWarning, match="No target network updater has been"
        ) if rl_warnings() else contextlib.nullcontext():
            loss_val = loss(**kwargs)
            loss_val_td = loss(td)

            torch.testing.assert_close(loss_val_td.get("loss_actor"), loss_val[0])
            torch.testing.assert_close(loss_val_td.get("loss_qvalue"), loss_val[1])
            torch.testing.assert_close(loss_val_td.get("loss_alpha"), loss_val[2])
            torch.testing.assert_close(loss_val_td.get("alpha"), loss_val[3])
            torch.testing.assert_close(loss_val_td.get("entropy"), loss_val[4])
            # test select
            torch.manual_seed(self.seed)
            loss.select_out_keys("loss_actor", "loss_alpha")
            if torch.__version__ >= "2.0.0":
                loss_actor, loss_alpha = loss(**kwargs)
            else:
                with pytest.raises(
                    RuntimeError,
                    match="You are likely using tensordict.nn.dispatch with keyword arguments",
                ):
                    loss_actor, loss_alpha = loss(**kwargs)
                return
            assert loss_actor == loss_val_td["loss_actor"]
            assert loss_alpha == loss_val_td["loss_alpha"]

    @pytest.mark.parametrize("action_key", ["action", "action2"])
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    def test_discrete_sac_terminating(
        self, action_key, observation_key, reward_key, done_key, terminated_key
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_sac(
            action_key=action_key,
            observation_key=observation_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
        )

        actor = self._create_mock_actor(
            observation_key=observation_key, action_key=action_key
        )
        qvalue = self._create_mock_qvalue(
            observation_key=observation_key,
        )

        loss = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=actor.spec[action_key].space.n,
            action_space="one-hot",
            skip_done_states=True,
        )
        loss.set_keys(
            action=action_key,
            reward=reward_key,
            done=done_key,
            terminated=terminated_key,
        )

        SoftUpdate(loss, eps=0.5)

        torch.manual_seed(0)
        done = td.get(("next", done_key))
        while not (done.any() and not done.all()):
            done = done.bernoulli_(0.1)
        obs_none = td.get(("next", observation_key))
        obs_none[done.squeeze(-1)] = float("nan")
        kwargs = {
            action_key: td.get(action_key),
            observation_key: td.get(observation_key),
            f"next_{reward_key}": td.get(("next", reward_key)),
            f"next_{done_key}": done,
            f"next_{terminated_key}": td.get(("next", terminated_key)),
            f"next_{observation_key}": obs_none,
        }
        td = TensorDict(kwargs, td.batch_size).unflatten_keys("_")
        assert loss(td).isfinite().all()

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_discrete_sac_reduction(self, reduction):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        td = self._create_mock_data_sac(device=device)
        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        loss_fn = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=actor.spec["action"].space.n,
            loss_function="l2",
            action_space="one-hot",
            delay_qvalue=False,
            reduction=reduction,
            scalar_output_mode="exclude" if reduction == "none" else None,
        )
        loss_fn.make_value_estimator()
        loss = loss_fn(td)
        if reduction == "none":
            for key in loss.keys():
                if key.startswith("loss"):
                    assert loss[key].shape == td.shape
        else:
            for key in loss.keys():
                if not key.startswith("loss"):
                    continue
                assert loss[key].shape == torch.Size([])


class TestCrossQ(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        observation_key="observation",
        action_key="action",
    ):
        # Actor
        action_spec = Bounded(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        net = nn.Sequential(nn.Linear(obs_dim, 2 * action_dim), NormalParamExtractor())
        module = TensorDictModule(
            net, in_keys=[observation_key], out_keys=["loc", "scale"]
        )
        actor = ProbabilisticActor(
            module=module,
            in_keys=["loc", "scale"],
            spec=action_spec,
            distribution_class=TanhNormal,
            out_keys=[action_key],
        )
        return actor.to(device)

    def _create_mock_qvalue(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        observation_key="observation",
        action_key="action",
        out_keys=None,
    ):
        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(obs_dim + action_dim, 1)

            def forward(self, obs, act):
                return self.linear(torch.cat([obs, act], -1))

        module = ValueClass()
        qvalue = ValueOperator(
            module=module,
            in_keys=[observation_key, action_key],
            out_keys=out_keys,
        )
        return qvalue.to(device)

    def _create_mock_common_layer_setup(
        self, n_obs=3, n_act=4, ncells=4, batch=2, n_hidden=2
    ):
        common = MLP(
            num_cells=ncells,
            in_features=n_obs,
            depth=3,
            out_features=n_hidden,
        )
        actor_net = MLP(
            num_cells=ncells,
            in_features=n_hidden,
            depth=1,
            out_features=2 * n_act,
        )
        qvalue = MLP(
            in_features=n_hidden + n_act,
            num_cells=ncells,
            depth=1,
            out_features=1,
        )
        batch = [batch]
        td = TensorDict(
            {
                "obs": torch.randn(*batch, n_obs),
                "action": torch.randn(*batch, n_act),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
                "terminated": torch.zeros(*batch, 1, dtype=torch.bool),
                "next": {
                    "obs": torch.randn(*batch, n_obs),
                    "reward": torch.randn(*batch, 1),
                    "done": torch.zeros(*batch, 1, dtype=torch.bool),
                    "terminated": torch.zeros(*batch, 1, dtype=torch.bool),
                },
            },
            batch,
        )
        common = Mod(common, in_keys=["obs"], out_keys=["hidden"])
        actor = ProbSeq(
            common,
            Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
            Mod(NormalParamExtractor(), in_keys=["param"], out_keys=["loc", "scale"]),
            ProbMod(
                in_keys=["loc", "scale"],
                out_keys=["action"],
                distribution_class=TanhNormal,
            ),
        )
        qvalue_head = Mod(
            qvalue, in_keys=["hidden", "action"], out_keys=["state_action_value"]
        )
        qvalue = Seq(common, qvalue_head)
        return actor, qvalue, common, td

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_crossq(
        self,
        batch=16,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
        observation_key="observation",
        action_key="action",
        done_key="done",
        terminated_key="terminated",
        reward_key="reward",
    ):
        # create a tensordict
        obs = torch.randn(batch, obs_dim, device=device)
        next_obs = torch.randn(batch, obs_dim, device=device)
        if atoms:
            raise NotImplementedError
        else:
            action = torch.randn(batch, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, 1, device=device)
        done = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch,),
            source={
                observation_key: obs,
                "next": {
                    observation_key: next_obs,
                    done_key: done,
                    terminated_key: terminated,
                    reward_key: reward,
                },
                action_key: action,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_crossq(
        self, batch=8, T=4, obs_dim=3, action_dim=4, atoms=None, device="cpu"
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim, device=device)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action = torch.randn(batch, T, atoms, action_dim, device=device).clamp(
                -1, 1
            )
        else:
            action = torch.randn(batch, T, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = torch.ones(batch, T, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "next": {
                    "observation": next_obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                    "done": done,
                    "terminated": terminated,
                    "reward": reward.masked_fill_(~mask.unsqueeze(-1), 0.0),
                },
                "collector": {"mask": mask},
                "action": action.masked_fill_(~mask.unsqueeze(-1), 0.0),
            },
            names=[None, "time"],
            device=device,
        )
        return td

    def test_reset_parameters_recursive(self):
        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()
        loss_fn = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
        )
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_crossq(
        self,
        num_qvalue,
        device,
        td_est,
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_crossq(device=device)
        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
        )

        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)

        with _check_td_steady(td):
            loss = loss_fn(td)

        assert loss_fn.tensor_keys.priority in td.keys()

        # check that losses are independent
        for k in loss.keys():
            if not k.startswith("loss"):
                continue
            loss[k].sum().backward(retain_graph=True)
            if k == "loss_actor":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            elif k == "loss_qvalue":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            elif k == "loss_alpha":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()

        sum(
            [item for name, item in loss.items() if name.startswith("loss_")]
        ).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            if not name.startswith("target_"):
                assert (
                    p.grad is not None and p.grad.norm() > 0.0
                ), f"parameter {name} (shape: {p.shape}) has a null gradient"
            else:
                assert (
                    p.grad is None or p.grad.norm() == 0.0
                ), f"target parameter {name} (shape: {p.shape}) has a non-null gradient"

    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_crossq_deactivate_vmap(
        self,
        num_qvalue,
        device,
        td_est,
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_crossq(device=device)
        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        torch.manual_seed(0)
        loss_fn_vmap = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            deactivate_vmap=False,
        )

        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn_vmap.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn_vmap.make_value_estimator(td_est)

        tdc = td.clone()
        with _check_td_steady(td):
            torch.manual_seed(1)
            loss_vmap = loss_fn_vmap(td)

        td = tdc

        torch.manual_seed(0)
        loss_fn_no_vmap = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            deactivate_vmap=True,
        )

        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn_no_vmap.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn_no_vmap.make_value_estimator(td_est)

        with pytest.raises(
            NotImplementedError,
            match="This implementation is not supported for torch<2.7",
        ) if torch.__version__ < "2.7" else contextlib.nullcontext():
            with _check_td_steady(td):
                torch.manual_seed(1)
                loss_no_vmap = loss_fn_no_vmap(td)
            assert_allclose_td(loss_vmap, loss_no_vmap)

    @pytest.mark.parametrize("num_qvalue", [2])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_crossq_state_dict(
        self,
        num_qvalue,
        device,
    ):
        torch.manual_seed(self.seed)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
        )
        sd = loss_fn.state_dict()
        loss_fn2 = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
        )
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("separate_losses", [False, True])
    def test_crossq_separate_losses(
        self,
        separate_losses,
        device,
    ):
        n_act = 4
        torch.manual_seed(self.seed)
        actor, qvalue, common, td = self._create_mock_common_layer_setup(n_act=n_act)

        loss_fn = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            action_spec=Unbounded(shape=(n_act,)),
            num_qvalue_nets=1,
            separate_losses=separate_losses,
        )
        loss = loss_fn(td)

        assert loss_fn.tensor_keys.priority in td.keys()

        # check that losses are independent
        for k in loss.keys():
            if not k.startswith("loss"):
                continue
            loss[k].sum().backward(retain_graph=True)
            if k == "loss_actor":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            elif k == "loss_qvalue":
                common_layers_no = len(list(common.parameters()))
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                if separate_losses:
                    common_layers = itertools.islice(
                        loss_fn.qvalue_network_params.values(True, True),
                        common_layers_no,
                    )
                    assert all(
                        (p.grad is None) or (p.grad == 0).all() for p in common_layers
                    )
                    qvalue_layers = itertools.islice(
                        loss_fn.qvalue_network_params.values(True, True),
                        common_layers_no,
                        None,
                    )
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all() for p in qvalue_layers
                    )
                else:
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.qvalue_network_params.values(True, True)
                    )
            elif k == "loss_alpha":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()

    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_crossq_batcher(
        self,
        n,
        num_qvalue,
        device,
    ):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_crossq(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
        )

        ms = MultiStep(gamma=0.9, n_steps=n).to(device)

        td_clone = td.clone()
        ms_td = ms(td_clone)

        torch.manual_seed(0)
        np.random.seed(0)

        with _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)
        assert loss_fn.tensor_keys.priority in ms_td.keys()

        with torch.no_grad():
            torch.manual_seed(0)  # log-prob is computed with a random action
            np.random.seed(0)
            loss = loss_fn(td)
        if n == 1:
            assert_allclose_td(td, ms_td.select(*list(td.keys(True, True))))
            _loss = sum(
                [item for name, item in loss.items() if name.startswith("loss_")]
            )
            _loss_ms = sum(
                [item for name, item in loss_ms.items() if name.startswith("loss_")]
            )
            assert (
                abs(_loss - _loss_ms) < 1e-3
            ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                assert_allclose_td(loss, loss_ms)
        sum(
            [item for name, item in loss_ms.items() if name.startswith("loss_")]
        ).backward()
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if not name.startswith("target_"):
                assert (
                    p.grad is not None and p.grad.norm() > 0.0
                ), f"parameter {name} (shape: {p.shape}) has a null gradient"
            else:
                assert (
                    p.grad is None or p.grad.norm() == 0.0
                ), f"target parameter {name} (shape: {p.shape}) has a non-null gradient"

        # Check param update effect on targets
        target_actor = [
            p.clone()
            for p in loss_fn.target_actor_network_params.values(
                include_nested=True, leaves_only=True
            )
        ]
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        target_actor2 = [
            p.clone()
            for p in loss_fn.target_actor_network_params.values(
                include_nested=True, leaves_only=True
            )
        ]

        assert not any((p1 == p2).any() for p1, p2 in zip(target_actor, target_actor2))

        # check that policy is updated after parameter update
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_crossq_tensordict_keys(self, td_est):

        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()

        loss_fn = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=2,
            loss_function="l2",
        )

        default_keys = {
            "priority": "td_error",
            "state_action_value": "state_action_value",
            "action": "action",
            "log_prob": "_log_prob",
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
        }

        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
            td_est=td_est,
        )

        qvalue = self._create_mock_qvalue()
        loss_fn = CrossQLoss(
            actor,
            qvalue,
            loss_function="l2",
        )

        key_mapping = {
            "reward": ("reward", "reward_test"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize("action_key", ["action", "action2"])
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    def test_crossq_notensordict(
        self, action_key, observation_key, reward_key, done_key, terminated_key
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_crossq(
            action_key=action_key,
            observation_key=observation_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
        )

        actor = self._create_mock_actor(
            observation_key=observation_key, action_key=action_key
        )
        qvalue = self._create_mock_qvalue(
            observation_key=observation_key,
            action_key=action_key,
            out_keys=["state_action_value"],
        )

        loss = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
        )
        loss.set_keys(
            action=action_key,
            reward=reward_key,
            done=done_key,
            terminated=terminated_key,
        )

        kwargs = {
            action_key: td.get(action_key),
            observation_key: td.get(observation_key),
            f"next_{reward_key}": td.get(("next", reward_key)),
            f"next_{done_key}": td.get(("next", done_key)),
            f"next_{terminated_key}": td.get(("next", terminated_key)),
            f"next_{observation_key}": td.get(("next", observation_key)),
        }
        td = TensorDict(kwargs, td.batch_size).unflatten_keys("_")

        # setting the seed for each loss so that drawing the random samples from value network
        # leads to same numbers for both runs
        torch.manual_seed(self.seed)
        loss_val = loss(**kwargs)

        torch.manual_seed(self.seed)

        loss_val_td = loss(td)
        assert len(loss_val) == 5

        torch.testing.assert_close(loss_val_td.get("loss_actor"), loss_val[0])
        torch.testing.assert_close(loss_val_td.get("loss_qvalue"), loss_val[1])
        torch.testing.assert_close(loss_val_td.get("loss_alpha"), loss_val[2])
        torch.testing.assert_close(loss_val_td.get("alpha"), loss_val[3])
        torch.testing.assert_close(loss_val_td.get("entropy"), loss_val[4])

        # test select
        torch.manual_seed(self.seed)
        loss.select_out_keys("loss_actor", "loss_alpha")
        if torch.__version__ >= "2.0.0":
            loss_actor, loss_alpha = loss(**kwargs)
        else:
            with pytest.raises(
                RuntimeError,
                match="You are likely using tensordict.nn.dispatch with keyword arguments",
            ):
                loss_actor, loss_alpha = loss(**kwargs)
            return
        assert loss_actor == loss_val_td["loss_actor"]
        assert loss_alpha == loss_val_td["loss_alpha"]

    def test_state_dict(
        self,
    ):

        model = torch.nn.Linear(3, 4)
        actor_module = TensorDictModule(model, in_keys=["obs"], out_keys=["logits"])
        policy = ProbabilisticActor(
            module=actor_module,
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=TanhDelta,
        )
        value = ValueOperator(module=model, in_keys=["obs"], out_keys="value")

        loss = CrossQLoss(
            actor_network=policy,
            qvalue_network=value,
            action_spec=Unbounded(shape=(2,)),
        )
        state = loss.state_dict()

        loss = CrossQLoss(
            actor_network=policy,
            qvalue_network=value,
            action_spec=Unbounded(shape=(2,)),
        )
        loss.load_state_dict(state)

        # with an access in between
        loss = CrossQLoss(
            actor_network=policy,
            qvalue_network=value,
            action_spec=Unbounded(shape=(2,)),
        )
        loss.target_entropy
        state = loss.state_dict()

        loss = CrossQLoss(
            actor_network=policy,
            qvalue_network=value,
            action_spec=Unbounded(shape=(2,)),
        )
        loss.load_state_dict(state)

    @pytest.mark.parametrize("action_dim", [1, 2, 4, 8])
    def test_crossq_target_entropy_auto(self, action_dim):
        """Regression test for target_entropy='auto' should be -dim(A)."""
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(action_dim=action_dim)
        qvalue = self._create_mock_qvalue(action_dim=action_dim)

        loss_fn = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
        )
        # target_entropy="auto" should compute -action_dim
        assert (
            loss_fn.target_entropy.item() == -action_dim
        ), f"target_entropy should be -{action_dim}, got {loss_fn.target_entropy.item()}"

    @pytest.mark.parametrize("target_entropy", [-1.0, -2.0, -5.0, 0.0])
    def test_crossq_target_entropy_explicit(self, target_entropy):
        """Regression test for issue #3309: explicit target_entropy should work."""
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()

        loss_fn = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            target_entropy=target_entropy,
        )
        assert (
            loss_fn.target_entropy.item() == target_entropy
        ), f"target_entropy should be {target_entropy}, got {loss_fn.target_entropy.item()}"

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_crossq_reduction(self, reduction):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        td = self._create_mock_data_crossq(device=device)
        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            loss_function="l2",
            reduction=reduction,
            scalar_output_mode="exclude" if reduction == "none" else None,
        )
        loss_fn.make_value_estimator()
        loss = loss_fn(td)
        if reduction == "none":
            for key in loss.keys():
                if key.startswith("loss"):
                    assert loss[key].shape == td.shape
        else:
            for key in loss.keys():
                if not key.startswith("loss"):
                    continue
                assert loss[key].shape == torch.Size([])


@pytest.mark.skipif(
    not _has_functorch, reason=f"functorch not installed: {FUNCTORCH_ERR}"
)
class TestREDQ(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        observation_key="observation",
        action_key="action",
    ):
        # Actor
        action_spec = Bounded(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        net = nn.Sequential(nn.Linear(obs_dim, 2 * action_dim), NormalParamExtractor())
        module = TensorDictModule(
            net, in_keys=[observation_key], out_keys=["loc", "scale"]
        )
        actor = ProbabilisticActor(
            module=module,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            return_log_prob=True,
            spec=action_spec,
            out_keys=[action_key],
        )
        return actor.to(device)

    def _create_mock_qvalue(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        observation_key="observation",
        action_key="action",
        out_keys=None,
    ):
        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(obs_dim + action_dim, 1)

            def forward(self, obs, act):
                return self.linear(torch.cat([obs, act], -1))

        module = ValueClass()
        qvalue = ValueOperator(
            module=module, in_keys=[observation_key, action_key], out_keys=out_keys
        )
        return qvalue.to(device)

    def _create_mock_common_layer_setup(
        self, n_obs=3, n_act=4, ncells=4, batch=2, n_hidden=2
    ):
        common = MLP(
            num_cells=ncells,
            in_features=n_obs,
            depth=3,
            out_features=n_hidden,
        )
        actor_net = MLP(
            num_cells=ncells,
            in_features=n_hidden,
            depth=1,
            out_features=2 * n_act,
        )
        value = MLP(
            in_features=n_hidden + n_act,
            num_cells=ncells,
            depth=1,
            out_features=1,
        )
        batch = [batch]
        td = TensorDict(
            {
                "obs": torch.randn(*batch, n_obs),
                "action": torch.randn(*batch, n_act),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
                "terminated": torch.zeros(*batch, 1, dtype=torch.bool),
                "next": {
                    "obs": torch.randn(*batch, n_obs),
                    "reward": torch.randn(*batch, 1),
                    "done": torch.zeros(*batch, 1, dtype=torch.bool),
                    "terminated": torch.zeros(*batch, 1, dtype=torch.bool),
                },
            },
            batch,
        )
        common = Mod(common, in_keys=["obs"], out_keys=["hidden"])
        actor = ProbSeq(
            common,
            Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
            Mod(NormalParamExtractor(), in_keys=["param"], out_keys=["loc", "scale"]),
            ProbMod(
                in_keys=["loc", "scale"],
                out_keys=["action"],
                distribution_class=TanhNormal,
                return_log_prob=True,
            ),
        )
        value_head = Mod(
            value, in_keys=["hidden", "action"], out_keys=["state_action_value"]
        )
        value = Seq(common, value_head)
        value(actor(td))
        return actor, value, common, td

    def _create_shared_mock_actor_qvalue(
        self, batch=2, obs_dim=3, action_dim=4, hidden_dim=5, device="cpu"
    ):
        class CommonClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(obs_dim, hidden_dim)

            def forward(self, obs):
                return self.linear(obs)

        class ActorClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Sequential(
                    nn.Linear(hidden_dim, 2 * action_dim), NormalParamExtractor()
                )

            def forward(self, hidden):
                return self.linear(hidden)

        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(hidden_dim + action_dim, 1)

            def forward(self, hidden, act):
                return self.linear(torch.cat([hidden, act], -1))

        common = TensorDictModule(
            CommonClass(), in_keys=["observation"], out_keys=["hidden"]
        )
        actor_subnet = ProbabilisticActor(
            TensorDictModule(
                ActorClass(), in_keys=["hidden"], out_keys=["loc", "scale"]
            ),
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            return_log_prob=True,
        )
        qvalue_subnet = ValueOperator(ValueClass(), in_keys=["hidden", "action"])
        model = ActorCriticOperator(common, actor_subnet, qvalue_subnet)
        return model.to(device)

    def _create_mock_data_redq(
        self,
        batch=16,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
        observation_key="observation",
        action_key="action",
        reward_key="reward",
        done_key="done",
        terminated_key="terminated",
    ):
        # create a tensordict
        obs = torch.randn(batch, obs_dim, device=device)
        next_obs = torch.randn(batch, obs_dim, device=device)
        if atoms:
            raise NotImplementedError
        else:
            action = torch.randn(batch, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, 1, device=device)
        done = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch,),
            source={
                observation_key: obs,
                "next": {
                    observation_key: next_obs,
                    done_key: done,
                    terminated_key: terminated,
                    reward_key: reward,
                },
                action_key: action,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_redq(
        self, batch=8, T=4, obs_dim=3, action_dim=4, atoms=None, device="cpu"
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim, device=device)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action = torch.randn(batch, T, atoms, action_dim, device=device).clamp(
                -1, 1
            )
        else:
            action = torch.randn(batch, T, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = ~torch.zeros(batch, T, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "next": {
                    "observation": next_obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                    "done": done,
                    "terminated": terminated,
                    "reward": reward.masked_fill_(~mask.unsqueeze(-1), 0.0),
                },
                "collector": {"mask": mask},
                "action": action.masked_fill_(~mask.unsqueeze(-1), 0.0),
            },
            names=[None, "time"],
            device=device,
        )
        return td

    def test_reset_parameters_recursive(self):
        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()
        loss_fn = REDQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
        )
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_redq(self, delay_qvalue, num_qvalue, device, td_est):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_redq(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = REDQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            delay_qvalue=delay_qvalue,
        )
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)

        with (
            pytest.warns(
                UserWarning,
                match="No target network updater has been associated with this loss module",
            )
            if delay_qvalue and rl_warnings()
            else contextlib.nullcontext()
        ):
            with _check_td_steady(td):
                loss = loss_fn(td)

            # check td is left untouched
            assert loss_fn.tensor_keys.priority in td.keys()

            # check that losses are independent
            for k in loss.keys():
                if not k.startswith("loss"):
                    continue
                loss[k].sum().backward(retain_graph=True)
                if k == "loss_actor":
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.qvalue_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                elif k == "loss_qvalue":
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.qvalue_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                elif k == "loss_alpha":
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.qvalue_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                else:
                    raise NotImplementedError(k)
                loss_fn.zero_grad()

            sum(
                [item for name, item in loss.items() if name.startswith("loss_")]
            ).backward()
            named_parameters = list(loss_fn.named_parameters())
            named_buffers = list(loss_fn.named_buffers())

            assert len({p for n, p in named_parameters}) == len(list(named_parameters))
            assert len({p for n, p in named_buffers}) == len(list(named_buffers))

            for name, p in named_parameters:
                if not name.startswith("target_"):
                    assert (
                        p.grad is not None and p.grad.norm() > 0.0
                    ), f"parameter {name} (shape: {p.shape}) has a null gradient"
                else:
                    assert (
                        p.grad is None or p.grad.norm() == 0.0
                    ), f"target parameter {name} (shape: {p.shape}) has a non-null gradient"

    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [2])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_redq_state_dict(self, delay_qvalue, num_qvalue, device):
        torch.manual_seed(self.seed)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = REDQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            delay_qvalue=delay_qvalue,
        )
        sd = loss_fn.state_dict()
        loss_fn2 = REDQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            delay_qvalue=delay_qvalue,
        )
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("action_dim", [1, 2, 4, 8])
    def test_redq_target_entropy_auto(self, action_dim):
        """Regression test for target_entropy='auto' should be -dim(A)."""
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(action_dim=action_dim)
        qvalue = self._create_mock_qvalue(action_dim=action_dim)

        loss_fn = REDQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
        )
        # target_entropy="auto" should compute -action_dim
        assert (
            loss_fn.target_entropy.item() == -action_dim
        ), f"target_entropy should be -{action_dim}, got {loss_fn.target_entropy.item()}"

    @pytest.mark.parametrize("separate_losses", [False, True])
    def test_redq_separate_losses(self, separate_losses):
        torch.manual_seed(self.seed)

        actor, qvalue, common, td = self._create_mock_common_layer_setup()

        loss_fn = REDQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            loss_function="l2",
            target_entropy=0.0,
            separate_losses=separate_losses,
        )

        with pytest.warns(
            UserWarning,
            match="No target network updater has been associated with this loss module",
        ) if rl_warnings() else contextlib.nullcontext():
            loss = loss_fn(td)

            # check that losses are independent
            for k in loss.keys():
                if not k.startswith("loss"):
                    continue
                loss[k].sum().backward(retain_graph=True)
                if k == "loss_actor":
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.qvalue_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                elif k == "loss_qvalue":
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                    if separate_losses:
                        common_layers_no = len(list(common.parameters()))
                        common_layers = itertools.islice(
                            loss_fn.qvalue_network_params.values(True, True),
                            common_layers_no,
                        )
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in common_layers
                        )
                        qvalue_layers = itertools.islice(
                            loss_fn.qvalue_network_params.values(True, True),
                            common_layers_no,
                            None,
                        )
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in qvalue_layers
                        )
                    else:
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in loss_fn.qvalue_network_params.values(
                                include_nested=True, leaves_only=True
                            )
                        )
                elif k == "loss_alpha":
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.qvalue_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                else:
                    raise NotImplementedError(k)
                loss_fn.zero_grad()

    @pytest.mark.parametrize("separate_losses", [False, True])
    def test_redq_deprecated_separate_losses(self, separate_losses):
        torch.manual_seed(self.seed)

        actor, qvalue, common, td = self._create_mock_common_layer_setup()

        loss_fn = REDQLoss_deprecated(
            actor_network=actor,
            qvalue_network=qvalue,
            loss_function="l2",
            target_entropy=0.0,
            separate_losses=separate_losses,
        )

        with pytest.warns(
            UserWarning,
            match="No target network updater has been associated with this loss module",
        ) if rl_warnings() else contextlib.nullcontext():
            loss = loss_fn(td)

        SoftUpdate(loss_fn, eps=0.5)

        # check that losses are independent
        for k in loss.keys():
            if not k.startswith("loss"):
                continue
            loss[k].sum().backward(retain_graph=True)
            if k == "loss_actor":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            elif k == "loss_qvalue":
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                if separate_losses:
                    common_layers_no = len(list(common.parameters()))
                    common_layers = itertools.islice(
                        loss_fn.qvalue_network_params.values(True, True),
                        common_layers_no,
                    )
                    assert all(
                        (p.grad is None) or (p.grad == 0).all() for p in common_layers
                    )
                    qvalue_layers = itertools.islice(
                        loss_fn.qvalue_network_params.values(True, True),
                        common_layers_no,
                        None,
                    )
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all() for p in qvalue_layers
                    )
                else:
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.qvalue_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
            elif k == "loss_alpha":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()

    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_redq_shared(self, delay_qvalue, num_qvalue, device):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_redq(device=device)

        actor_critic = self._create_shared_mock_actor_qvalue(device=device)
        actor = actor_critic.get_policy_operator()
        qvalue = actor_critic.get_critic_operator()

        loss_fn = REDQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            delay_qvalue=delay_qvalue,
            target_entropy=0.0,
        )

        if delay_qvalue:
            target_updater = SoftUpdate(loss_fn, tau=0.05)

        with _check_td_steady(td):
            loss = loss_fn(td)

        # check that losses are independent
        for k in loss.keys():
            if not k.startswith("loss"):
                continue
            loss[k].sum().backward(retain_graph=True)
            if k == "loss_actor":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(True, True)
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(True, True)
                )
            elif k == "loss_qvalue":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(True, True)
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(True, True)
                    if isinstance(p, nn.Parameter)
                )
            elif k == "loss_alpha":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(True, True)
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(True, True)
                )
            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()

    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_redq_batched(self, delay_qvalue, num_qvalue, device, td_est):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_redq(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = REDQLoss(
            actor_network=deepcopy(actor),
            qvalue_network=deepcopy(qvalue),
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            delay_qvalue=delay_qvalue,
        )
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)

        loss_class_deprec = (
            REDQLoss_deprecated if not delay_qvalue else DoubleREDQLoss_deprecated
        )
        loss_fn_deprec = loss_class_deprec(
            actor_network=deepcopy(actor),
            qvalue_network=deepcopy(qvalue),
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            delay_qvalue=delay_qvalue,
        )
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn_deprec.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn_deprec.make_value_estimator(td_est)

        td_clone1 = td.clone()
        td_clone2 = td.clone()
        torch.manual_seed(0)
        with (
            pytest.warns(UserWarning, match="No target network updater has been")
            if delay_qvalue and rl_warnings()
            else contextlib.nullcontext()
        ):
            with _check_td_steady(td_clone1):
                loss_fn(td_clone1)

            torch.manual_seed(0)
            with _check_td_steady(td_clone2):
                loss_fn_deprec(td_clone2)

        # TODO: find a way to compare the losses: problem is that we sample actions either sequentially or in batch,
        #  so setting seed has little impact

    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_redq_batcher(self, n, delay_qvalue, num_qvalue, device, gamma=0.9):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_redq(device=device)
        assert td.names == td.get("next").names

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = REDQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            delay_qvalue=delay_qvalue,
        )

        ms = MultiStep(gamma=gamma, n_steps=n).to(device)

        td_clone = td.clone()
        assert td_clone.names == td_clone.get("next").names
        ms_td = ms(td_clone)
        assert ms_td.names == ms_td.get("next").names

        torch.manual_seed(0)
        np.random.seed(0)

        with (
            pytest.warns(UserWarning, match="No target network updater has been")
            if delay_qvalue and rl_warnings()
            else contextlib.nullcontext()
        ):
            with _check_td_steady(ms_td):
                loss_ms = loss_fn(ms_td)
            assert loss_fn.tensor_keys.priority in ms_td.keys()

            with torch.no_grad():
                torch.manual_seed(0)  # log-prob is computed with a random action
                np.random.seed(0)
                loss = loss_fn(td)
            if n == 1:
                assert_allclose_td(td, ms_td.select(*list(td.keys(True, True))))
                _loss = sum(
                    [item for name, item in loss.items() if name.startswith("loss_")]
                )
                _loss_ms = sum(
                    [item for name, item in loss_ms.items() if name.startswith("loss_")]
                )
                assert (
                    abs(_loss - _loss_ms) < 1e-3
                ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
            else:
                with pytest.raises(AssertionError):
                    assert_allclose_td(loss, loss_ms)
            sum(
                [item for name, item in loss_ms.items() if name.startswith("loss_")]
            ).backward()
            named_parameters = loss_fn.named_parameters()
            for name, p in named_parameters:
                if not name.startswith("target_"):
                    assert (
                        p.grad is not None and p.grad.norm() > 0.0
                    ), f"parameter {name} (shape: {p.shape}) has a null gradient"
                else:
                    assert (
                        p.grad is None or p.grad.norm() == 0.0
                    ), f"target parameter {name} (shape: {p.shape}) has a non-null gradient"

            # Check param update effect on targets
            target_actor = loss_fn.target_actor_network_params.clone().values(
                include_nested=True, leaves_only=True
            )
            target_qvalue = loss_fn.target_qvalue_network_params.clone().values(
                include_nested=True, leaves_only=True
            )
            for p in loss_fn.parameters():
                if p.requires_grad:
                    p.data += torch.randn_like(p)
            target_actor2 = loss_fn.target_actor_network_params.clone().values(
                include_nested=True, leaves_only=True
            )
            target_qvalue2 = loss_fn.target_qvalue_network_params.clone().values(
                include_nested=True, leaves_only=True
            )
            if loss_fn.delay_actor:
                assert all(
                    (p1 == p2).all() for p1, p2 in zip(target_actor, target_actor2)
                )
            else:
                assert not any(
                    (p1 == p2).any() for p1, p2 in zip(target_actor, target_actor2)
                )
            if loss_fn.delay_qvalue:
                assert all(
                    (p1 == p2).all() for p1, p2 in zip(target_qvalue, target_qvalue2)
                )
            else:
                assert not any(
                    (p1 == p2).any() for p1, p2 in zip(target_qvalue, target_qvalue2)
                )

            # check that policy is updated after parameter update
            actorp_set = set(actor.parameters())
            loss_fnp_set = set(loss_fn.parameters())
            assert len(actorp_set.intersection(loss_fnp_set)) == len(actorp_set)
            parameters = [p.clone() for p in actor.parameters()]
            for p in loss_fn.parameters():
                if p.requires_grad:
                    p.data += torch.randn_like(p)
            assert all(
                (p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters())
            )

    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_redq_tensordict_keys(self, td_est):
        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()

        loss_fn = REDQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            loss_function="l2",
        )

        default_keys = {
            "priority": "td_error",
            "action": "action",
            "value": "state_value",
            "sample_log_prob": "action_log_prob",
            "state_action_value": "state_action_value",
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
        }
        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
            td_est=td_est,
        )

        qvalue = self._create_mock_qvalue()
        loss_fn = REDQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            loss_function="l2",
        )

        key_mapping = {
            "value": ("value", "state_value_test"),
            "reward": ("reward", "reward_test"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize("action_key", ["action", "action2"])
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    @pytest.mark.parametrize("deprec", [True, False])
    def test_redq_notensordict(
        self, action_key, observation_key, reward_key, done_key, terminated_key, deprec
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_redq(
            action_key=action_key,
            observation_key=observation_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
        )

        actor = self._create_mock_actor(
            observation_key=observation_key,
            action_key=action_key,
        )
        qvalue = self._create_mock_qvalue(
            observation_key=observation_key,
            action_key=action_key,
            out_keys=["state_action_value"],
        )

        if deprec:
            cls = REDQLoss_deprecated
        else:
            cls = REDQLoss
        loss = cls(
            actor_network=actor,
            qvalue_network=qvalue,
        )
        if deprec:
            loss.set_keys(
                action=action_key,
                reward=reward_key,
                done=done_key,
                terminated=terminated_key,
                log_prob=_add_suffix(action_key, "_log_prob"),
            )
        else:
            loss.set_keys(
                action=action_key,
                reward=reward_key,
                done=done_key,
                terminated=terminated_key,
                sample_log_prob=_add_suffix(action_key, "_log_prob"),
            )

        kwargs = {
            action_key: td.get(action_key),
            observation_key: td.get(observation_key),
            f"next_{reward_key}": td.get(("next", reward_key)),
            f"next_{done_key}": td.get(("next", done_key)),
            f"next_{terminated_key}": td.get(("next", terminated_key)),
            f"next_{observation_key}": td.get(("next", observation_key)),
        }
        td = TensorDict(kwargs, td.batch_size).unflatten_keys("_")

        torch.manual_seed(self.seed)
        with pytest.warns(
            UserWarning,
            match="No target network updater has been associated with this loss module",
        ) if rl_warnings() else contextlib.nullcontext():
            loss_val = loss(**kwargs)
            torch.manual_seed(self.seed)
            loss_val_td = loss(td)

            torch.testing.assert_close(loss_val_td.get("loss_actor"), loss_val[0])
            torch.testing.assert_close(loss_val_td.get("loss_qvalue"), loss_val[1])
            torch.testing.assert_close(loss_val_td.get("loss_alpha"), loss_val[2])
            torch.testing.assert_close(loss_val_td.get("alpha"), loss_val[3])
            torch.testing.assert_close(loss_val_td.get("entropy"), loss_val[4])
            if not deprec:
                torch.testing.assert_close(
                    loss_val_td.get("state_action_value_actor"), loss_val[5]
                )
                torch.testing.assert_close(
                    loss_val_td.get("action_log_prob_actor"), loss_val[6]
                )
                torch.testing.assert_close(
                    loss_val_td.get("next.state_value"), loss_val[7]
                )
                torch.testing.assert_close(loss_val_td.get("target_value"), loss_val[8])
            # test select
            torch.manual_seed(self.seed)
            loss.select_out_keys("loss_actor", "loss_alpha")
            if torch.__version__ >= "2.0.0":
                loss_actor, loss_alpha = loss(**kwargs)
            else:
                with pytest.raises(
                    RuntimeError,
                    match="You are likely using tensordict.nn.dispatch with keyword arguments",
                ):
                    loss_actor, loss_alpha = loss(**kwargs)
                return
            assert loss_actor == loss_val_td["loss_actor"]
            assert loss_alpha == loss_val_td["loss_alpha"]

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    @pytest.mark.parametrize("deprecated_loss", [True, False])
    def test_redq_reduction(self, reduction, deprecated_loss):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        td = self._create_mock_data_redq(device=device)
        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        if deprecated_loss:
            loss_fn = REDQLoss_deprecated(
                actor_network=actor,
                qvalue_network=qvalue,
                loss_function="l2",
                delay_qvalue=False,
                reduction=reduction,
            )
        else:
            loss_fn = REDQLoss(
                actor_network=actor,
                qvalue_network=qvalue,
                loss_function="l2",
                delay_qvalue=False,
                reduction=reduction,
                scalar_output_mode="exclude" if reduction == "none" else None,
            )
        loss_fn.make_value_estimator()
        loss = loss_fn(td)
        if reduction == "none":
            for key in loss.keys():
                if key.startswith("loss"):
                    assert loss[key].shape[-1] == td.shape[0]
        else:
            for key in loss.keys():
                if not key.startswith("loss"):
                    continue
                assert loss[key].shape == torch.Size([])


class TestCQL(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = Bounded(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        net = nn.Sequential(nn.Linear(obs_dim, 2 * action_dim), NormalParamExtractor())
        module = TensorDictModule(
            net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        actor = ProbabilisticActor(
            module=module,
            in_keys=["loc", "scale"],
            spec=action_spec,
            distribution_class=TanhNormal,
        )
        return actor.to(device)

    def _create_mock_qvalue(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(obs_dim + action_dim, 1)

            def forward(self, obs, act):
                return self.linear(torch.cat([obs, act], -1))

        module = ValueClass()
        qvalue = ValueOperator(
            module=module,
            in_keys=["observation", "action"],
        )
        return qvalue.to(device)

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_cql(
        self, batch=16, obs_dim=3, action_dim=4, atoms=None, device="cpu"
    ):
        # create a tensordict
        obs = torch.randn(batch, obs_dim, device=device)
        next_obs = torch.randn(batch, obs_dim, device=device)
        if atoms:
            raise NotImplementedError
        else:
            action = torch.randn(batch, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, 1, device=device)
        done = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next": {
                    "observation": next_obs,
                    "done": done,
                    "terminated": terminated,
                    "reward": reward,
                },
                "action": action,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_cql(
        self, batch=8, T=4, obs_dim=3, action_dim=4, atoms=None, device="cpu"
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim, device=device)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action = torch.randn(batch, T, atoms, action_dim, device=device).clamp(
                -1, 1
            )
        else:
            action = torch.randn(batch, T, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = torch.ones(batch, T, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "next": {
                    "observation": next_obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                    "done": done,
                    "terminated": terminated,
                    "reward": reward.masked_fill_(~mask.unsqueeze(-1), 0.0),
                },
                "collector": {"mask": mask},
                "action": action.masked_fill_(~mask.unsqueeze(-1), 0.0),
            },
            names=[None, "time"],
            device=device,
        )
        return td

    def test_reset_parameters_recursive(self):
        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()
        loss_fn = CQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
        )
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("delay_actor", (True, False))
    @pytest.mark.parametrize("delay_qvalue", (True, True))
    @pytest.mark.parametrize("max_q_backup", [True, False])
    @pytest.mark.parametrize("deterministic_backup", [True, False])
    @pytest.mark.parametrize("with_lagrange", [True, False])
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_cql(
        self,
        delay_actor,
        delay_qvalue,
        max_q_backup,
        deterministic_backup,
        with_lagrange,
        device,
        td_est,
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_cql(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = CQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            loss_function="l2",
            max_q_backup=max_q_backup,
            deterministic_backup=deterministic_backup,
            with_lagrange=with_lagrange,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
        )

        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return

        if td_est is not None:
            loss_fn.make_value_estimator(td_est)

        with _check_td_steady(td), pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
            loss = loss_fn(td)
        assert loss_fn.tensor_keys.priority in td.keys()

        # check that losses are independent
        for k in loss.keys():
            if not k.startswith("loss"):
                continue
            if k == "loss_alpha_prime" and not with_lagrange:
                continue
            loss[k].sum().backward(retain_graph=True)
            if k == "loss_actor":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            elif k == "loss_actor_bc":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            elif k == "loss_qvalue":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            elif k == "loss_cql":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert not all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            elif k == "loss_alpha":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            elif k == "loss_alpha_prime":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()
            assert all(
                (p.grad is None) or (p.grad == 0).all()
                for p in loss_fn.actor_network_params.values(
                    include_nested=True, leaves_only=True
                )
            )
            assert all(
                (p.grad is None) or (p.grad == 0).all()
                for p in loss_fn.qvalue_network_params.values(
                    include_nested=True, leaves_only=True
                )
            )

        sum(
            [item for name, item in loss.items() if name.startswith("loss_")]
        ).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            if not name.startswith("target_"):
                assert (
                    p.grad is not None and p.grad.norm() > 0.0
                ), f"parameter {name} (shape: {p.shape}) has a null gradient"
            else:
                assert (
                    p.grad is None or p.grad.norm() == 0.0
                ), f"target parameter {name} (shape: {p.shape}) has a non-null gradient"

    @pytest.mark.parametrize("delay_actor", (True,))
    @pytest.mark.parametrize("delay_qvalue", (True,))
    @pytest.mark.parametrize(
        "max_q_backup",
        [
            True,
        ],
    )
    @pytest.mark.parametrize(
        "deterministic_backup",
        [
            True,
        ],
    )
    @pytest.mark.parametrize(
        "with_lagrange",
        [
            True,
        ],
    )
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("td_est", [None])
    def test_cql_deactivate_vmap(
        self,
        delay_actor,
        delay_qvalue,
        max_q_backup,
        deterministic_backup,
        with_lagrange,
        device,
        td_est,
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_cql(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        torch.manual_seed(0)
        loss_fn_vmap = CQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            loss_function="l2",
            max_q_backup=max_q_backup,
            deterministic_backup=deterministic_backup,
            with_lagrange=with_lagrange,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
            deactivate_vmap=False,
        )

        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn_vmap.make_value_estimator(td_est)
            return

        if td_est is not None:
            loss_fn_vmap.make_value_estimator(td_est)
        tdc = td.clone()
        with _check_td_steady(td), pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
            torch.manual_seed(1)
            loss_vmap = loss_fn_vmap(td)
        td = tdc

        torch.manual_seed(0)
        loss_fn_no_vmap = CQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            loss_function="l2",
            max_q_backup=max_q_backup,
            deterministic_backup=deterministic_backup,
            with_lagrange=with_lagrange,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
            deactivate_vmap=True,
        )

        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn_no_vmap.make_value_estimator(td_est)
            return

        if td_est is not None:
            loss_fn_no_vmap.make_value_estimator(td_est)

        with pytest.raises(
            NotImplementedError,
            match="This implementation is not supported for torch<2.7",
        ) if torch.__version__ < "2.7" else contextlib.nullcontext():
            with _check_td_steady(td), pytest.warns(
                UserWarning, match="No target network updater"
            ) if rl_warnings() else contextlib.nullcontext():
                torch.manual_seed(1)
                loss_no_vmap = loss_fn_no_vmap(td)
            assert_allclose_td(loss_vmap, loss_no_vmap)

    @pytest.mark.parametrize("delay_actor", (True,))
    @pytest.mark.parametrize("delay_qvalue", (True,))
    @pytest.mark.parametrize("max_q_backup", [True])
    @pytest.mark.parametrize("deterministic_backup", [True])
    @pytest.mark.parametrize("with_lagrange", [True])
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("td_est", [None])
    def test_cql_qvalfromlist(
        self,
        delay_actor,
        delay_qvalue,
        max_q_backup,
        deterministic_backup,
        with_lagrange,
        device,
        td_est,
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_cql(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue0 = self._create_mock_qvalue(device=device)
        qvalue1 = self._create_mock_qvalue(device=device)

        loss_fn_single = CQLLoss(
            actor_network=actor,
            qvalue_network=qvalue0,
            loss_function="l2",
            max_q_backup=max_q_backup,
            deterministic_backup=deterministic_backup,
            with_lagrange=with_lagrange,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
        )
        loss_fn_mult = CQLLoss(
            actor_network=actor,
            qvalue_network=[qvalue0, qvalue1],
            loss_function="l2",
            max_q_backup=max_q_backup,
            deterministic_backup=deterministic_backup,
            with_lagrange=with_lagrange,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
        )
        # Check that all params have the same shape
        p2 = dict(loss_fn_mult.named_parameters())
        for key, val in loss_fn_single.named_parameters():
            assert val.shape == p2[key].shape
        assert len(dict(loss_fn_single.named_parameters())) == len(p2)

    @pytest.mark.parametrize("delay_actor", (True, False))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("max_q_backup", [True])
    @pytest.mark.parametrize("deterministic_backup", [True])
    @pytest.mark.parametrize("with_lagrange", [True])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_cql_state_dict(
        self,
        delay_actor,
        delay_qvalue,
        max_q_backup,
        deterministic_backup,
        with_lagrange,
        device,
    ):
        if delay_actor or delay_qvalue:
            pytest.skip("incompatible config")

        torch.manual_seed(self.seed)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        kwargs = {}
        if delay_actor:
            kwargs["delay_actor"] = True
        if delay_qvalue:
            kwargs["delay_qvalue"] = True

        loss_fn = CQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            loss_function="l2",
            max_q_backup=max_q_backup,
            deterministic_backup=deterministic_backup,
            with_lagrange=with_lagrange,
            **kwargs,
        )
        sd = loss_fn.state_dict()
        loss_fn2 = CQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            loss_function="l2",
            max_q_backup=max_q_backup,
            deterministic_backup=deterministic_backup,
            with_lagrange=with_lagrange,
            **kwargs,
        )
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("action_dim", [1, 2, 4, 8])
    def test_cql_target_entropy_auto(self, action_dim):
        """Regression test for target_entropy='auto' should be -dim(A)."""
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(action_dim=action_dim)
        qvalue = self._create_mock_qvalue(action_dim=action_dim)

        loss_fn = CQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
        )
        # target_entropy="auto" should compute -action_dim
        assert (
            loss_fn.target_entropy.item() == -action_dim
        ), f"target_entropy should be -{action_dim}, got {loss_fn.target_entropy.item()}"

    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("delay_actor", (True, False))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("max_q_backup", [True, False])
    @pytest.mark.parametrize("deterministic_backup", [True, False])
    @pytest.mark.parametrize("with_lagrange", [True, False])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_cql_batcher(
        self,
        n,
        delay_actor,
        delay_qvalue,
        max_q_backup,
        deterministic_backup,
        with_lagrange,
        device,
    ):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_cql(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        kwargs = {}
        if delay_actor:
            kwargs["delay_actor"] = True
        if delay_qvalue:
            kwargs["delay_qvalue"] = True

        loss_fn = CQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            loss_function="l2",
            max_q_backup=max_q_backup,
            deterministic_backup=deterministic_backup,
            with_lagrange=with_lagrange,
            **kwargs,
        )

        ms = MultiStep(gamma=0.9, n_steps=n).to(device)

        td_clone = td.clone()
        ms_td = ms(td_clone)

        torch.manual_seed(0)
        np.random.seed(0)
        with pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
            with _check_td_steady(ms_td):
                loss_ms = loss_fn(ms_td)
            assert loss_fn.tensor_keys.priority in ms_td.keys()

            with torch.no_grad():
                torch.manual_seed(0)  # log-prob is computed with a random action
                np.random.seed(0)
                loss = loss_fn(td)
            if n == 1:
                assert_allclose_td(td, ms_td.select(*list(td.keys(True, True))))
                _loss = sum(
                    [item for name, item in loss.items() if name.startswith("loss_")]
                )
                _loss_ms = sum(
                    [item for name, item in loss_ms.items() if name.startswith("loss_")]
                )
                assert (
                    abs(_loss - _loss_ms) < 1e-3
                ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
            else:
                with pytest.raises(AssertionError):
                    assert_allclose_td(loss, loss_ms)
            sum(
                [item for name, item in loss_ms.items() if name.startswith("loss_")]
            ).backward()
            named_parameters = loss_fn.named_parameters()
            for name, p in named_parameters:
                if not name.startswith("target_"):
                    assert (
                        p.grad is not None and p.grad.norm() > 0.0
                    ), f"parameter {name} (shape: {p.shape}) has a null gradient"
                else:
                    assert (
                        p.grad is None or p.grad.norm() == 0.0
                    ), f"target parameter {name} (shape: {p.shape}) has a non-null gradient"

            # Check param update effect on targets
            target_actor = [
                p.clone()
                for p in loss_fn.target_actor_network_params.values(
                    include_nested=True, leaves_only=True
                )
            ]
            target_qvalue = [
                p.clone()
                for p in loss_fn.target_qvalue_network_params.values(
                    include_nested=True, leaves_only=True
                )
            ]
            for p in loss_fn.parameters():
                if p.requires_grad:
                    p.data += torch.randn_like(p)
            target_actor2 = [
                p.clone()
                for p in loss_fn.target_actor_network_params.values(
                    include_nested=True, leaves_only=True
                )
            ]
            target_qvalue2 = [
                p.clone()
                for p in loss_fn.target_qvalue_network_params.values(
                    include_nested=True, leaves_only=True
                )
            ]
            if loss_fn.delay_actor:
                assert all(
                    (p1 == p2).all() for p1, p2 in zip(target_actor, target_actor2)
                )
            else:
                assert not any(
                    (p1 == p2).any() for p1, p2 in zip(target_actor, target_actor2)
                )
            if loss_fn.delay_qvalue:
                assert all(
                    (p1 == p2).all() for p1, p2 in zip(target_qvalue, target_qvalue2)
                )
            else:
                assert not any(
                    (p1 == p2).any() for p1, p2 in zip(target_qvalue, target_qvalue2)
                )

            # check that policy is updated after parameter update
            parameters = [p.clone() for p in actor.parameters()]
            for p in loss_fn.parameters():
                if p.requires_grad:
                    p.data += torch.randn_like(p)
            assert all(
                (p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters())
            )

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_cql_reduction(self, reduction):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        td = self._create_mock_data_cql(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = CQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            loss_function="l2",
            delay_actor=False,
            delay_qvalue=False,
            reduction=reduction,
            scalar_output_mode="exclude" if reduction == "none" else None,
        )
        loss_fn.make_value_estimator()
        loss = loss_fn(td)
        if reduction == "none":
            for key in loss.keys():
                if key.startswith("loss"):
                    assert loss[key].shape == td.shape
        else:
            for key in loss.keys():
                if not key.startswith("loss"):
                    continue
                assert loss[key].shape == torch.Size([])


class TestDiscreteCQL(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(
        self,
        action_spec_type,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        is_nn_module=False,
        action_value_key=None,
    ):
        # Actor
        if action_spec_type == "one_hot":
            action_spec = OneHot(action_dim)
        elif action_spec_type == "categorical":
            action_spec = Categorical(action_dim)
        else:
            raise ValueError(f"Wrong action spec type: {action_spec_type}")

        module = nn.Linear(obs_dim, action_dim)
        if is_nn_module:
            return module.to(device)
        actor = QValueActor(
            spec=Composite(
                {
                    "action": action_spec,
                    (
                        "action_value" if action_value_key is None else action_value_key
                    ): None,
                    "chosen_action_value": None,
                },
                shape=[],
            ),
            action_space=action_spec_type,
            module=module,
            action_value_key=action_value_key,
        ).to(device)
        return actor

    def _create_mock_data_dcql(
        self,
        action_spec_type,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        action_key="action",
        action_value_key="action_value",
    ):
        # create a tensordict
        obs = torch.randn(batch, obs_dim)
        next_obs = torch.randn(batch, obs_dim)

        action_value = torch.randn(batch, action_dim)
        action = (action_value == action_value.max(-1, True)[0]).to(torch.long)

        if action_spec_type == "categorical":
            action_value = torch.max(action_value, -1, keepdim=True)[0]
            action = torch.argmax(action, -1, keepdim=False)
        reward = torch.randn(batch, 1)
        done = torch.zeros(batch, 1, dtype=torch.bool)
        terminated = torch.zeros(batch, 1, dtype=torch.bool)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next": {
                    "observation": next_obs,
                    "done": done,
                    "terminated": terminated,
                    "reward": reward,
                },
                action_key: action,
                action_value_key: action_value,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_dcql(
        self,
        action_spec_type,
        batch=2,
        T=4,
        obs_dim=3,
        action_dim=4,
        device="cpu",
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim, device=device)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]

        action_value = torch.randn(batch, T, action_dim, device=device)
        action = (action_value == action_value.max(-1, True)[0]).to(torch.long)

        # action_value = action_value.unsqueeze(-1)
        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = ~torch.zeros(batch, T, dtype=torch.bool, device=device)
        if action_spec_type == "categorical":
            action_value = torch.max(action_value, -1, keepdim=True)[0]
            action = torch.argmax(action, -1, keepdim=False)
            action = action.masked_fill_(~mask, 0.0)
        else:
            action = action.masked_fill_(~mask.unsqueeze(-1), 0.0)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "next": {
                    "observation": next_obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                    "done": done,
                    "terminated": terminated,
                    "reward": reward.masked_fill_(~mask.unsqueeze(-1), 0.0),
                },
                "collector": {"mask": mask},
                "action": action,
                "action_value": action_value.masked_fill_(~mask.unsqueeze(-1), 0.0),
            },
            names=[None, "time"],
        )
        return td

    def test_reset_parameters_recursive(self):
        actor = self._create_mock_actor(
            action_spec_type="one_hot",
        )
        loss_fn = DiscreteCQLLoss(actor)
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("action_spec_type", ("one_hot", "categorical"))
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_dcql(self, delay_value, device, action_spec_type, td_est):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, device=device
        )
        td = self._create_mock_data_dcql(
            action_spec_type=action_spec_type, device=device
        )
        loss_fn = DiscreteCQLLoss(actor, loss_function="l2", delay_value=delay_value)
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)
        with (
            pytest.warns(UserWarning, match="No target network updater has been")
            if delay_value and rl_warnings()
            else contextlib.nullcontext()
        ), _check_td_steady(td):
            loss = loss_fn(td)
        assert loss_fn.tensor_keys.priority in td.keys(True)

        if delay_value:
            SoftUpdate(loss_fn, eps=0.5)

        sum([item for key, item in loss.items() if key.startswith("loss")]).backward()
        assert torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 1.0) > 0.0

        # Check param update effect on targets
        target_value = loss_fn.target_value_network_params.clone()
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        target_value2 = loss_fn.target_value_network_params.clone()
        if loss_fn.delay_value:
            assert_allclose_td(target_value, target_value2)
        else:
            assert not (target_value == target_value2).any()

        # check that policy is updated after parameter update
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("action_spec_type", ("one_hot", "categorical"))
    def test_dcql_state_dict(self, delay_value, device, action_spec_type):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, device=device
        )
        loss_fn = DiscreteCQLLoss(actor, loss_function="l2", delay_value=delay_value)
        sd = loss_fn.state_dict()
        loss_fn2 = DiscreteCQLLoss(actor, loss_function="l2", delay_value=delay_value)
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("action_spec_type", ("one_hot", "categorical"))
    def test_dcql_batcher(self, n, delay_value, device, action_spec_type, gamma=0.9):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, device=device
        )

        td = self._create_seq_mock_data_dcql(
            action_spec_type=action_spec_type, device=device
        )
        loss_fn = DiscreteCQLLoss(actor, loss_function="l2", delay_value=delay_value)

        ms = MultiStep(gamma=gamma, n_steps=n).to(device)
        ms_td = ms(td.clone())

        with (
            pytest.warns(UserWarning, match="No target network updater has been")
            if delay_value and rl_warnings()
            else contextlib.nullcontext()
        ), _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)
        assert loss_fn.tensor_keys.priority in ms_td.keys()

        if delay_value:
            SoftUpdate(loss_fn, eps=0.5)

        with torch.no_grad():
            loss = loss_fn(td)
        if n == 1:
            assert_allclose_td(td, ms_td.select(*td.keys(True, True)))
            _loss = sum([item for key, item in loss.items() if key.startswith("loss_")])
            _loss_ms = sum(
                [item for key, item in loss_ms.items() if key.startswith("loss_")]
            )
            assert (
                abs(_loss - _loss_ms) < 1e-3
            ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                assert_allclose_td(loss, loss_ms)
        sum(
            [item for key, item in loss_ms.items() if key.startswith("loss_")]
        ).backward()
        assert torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 1.0) > 0.0

        # Check param update effect on targets
        target_value = loss_fn.target_value_network_params.clone()
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        target_value2 = loss_fn.target_value_network_params.clone()
        if loss_fn.delay_value:
            assert_allclose_td(target_value, target_value2)
        else:
            assert not (target_value == target_value2).any()

        # check that policy is updated after parameter update
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_dcql_tensordict_keys(self, td_est):
        torch.manual_seed(self.seed)
        action_spec_type = "one_hot"
        actor = self._create_mock_actor(action_spec_type=action_spec_type)
        loss_fn = DQNLoss(actor, delay_value=True)

        default_keys = {
            "value_target": "value_target",
            "value": "chosen_action_value",
            "priority": "td_error",
            "action_value": "action_value",
            "action": "action",
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
        }

        self.tensordict_keys_test(loss_fn, default_keys=default_keys)

        loss_fn = DiscreteCQLLoss(actor)
        key_mapping = {
            "reward": ("reward", "reward_test"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, action_value_key="chosen_action_value_2"
        )
        loss_fn = DiscreteCQLLoss(actor)
        key_mapping = {
            "value": ("value", "chosen_action_value_2"),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize("action_spec_type", ("categorical", "one_hot"))
    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_dcql_tensordict_run(self, action_spec_type, td_est):
        torch.manual_seed(self.seed)
        tensor_keys = {
            "action_value": "action_value_test",
            "action": "action_test",
            "priority": "priority_test",
        }
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type,
            action_value_key=tensor_keys["action_value"],
        )
        td = self._create_mock_data_dcql(
            action_spec_type=action_spec_type,
            action_key=tensor_keys["action"],
            action_value_key=tensor_keys["action_value"],
        )

        loss_fn = DiscreteCQLLoss(actor, loss_function="l2")
        loss_fn.set_keys(**tensor_keys)

        SoftUpdate(loss_fn, eps=0.5)

        if td_est is not None:
            loss_fn.make_value_estimator(td_est)
        with _check_td_steady(td):
            _ = loss_fn(td)
        assert loss_fn.tensor_keys.priority in td.keys()

    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    def test_dcql_notensordict(
        self, observation_key, reward_key, done_key, terminated_key
    ):
        n_obs = 3
        n_action = 4
        action_spec = OneHot(n_action)
        module = nn.Linear(n_obs, n_action)  # a simple value model
        actor = QValueActor(
            spec=action_spec,
            action_space="one_hot",
            module=module,
            in_keys=[observation_key],
        )
        loss = DiscreteCQLLoss(actor)

        SoftUpdate(loss, eps=0.5)

        loss.set_keys(reward=reward_key, done=done_key, terminated=terminated_key)
        # define data
        observation = torch.randn(n_obs)
        next_observation = torch.randn(n_obs)
        action = action_spec.rand()
        next_reward = torch.randn(1)
        next_done = torch.zeros(1, dtype=torch.bool)
        next_terminated = torch.zeros(1, dtype=torch.bool)
        kwargs = {
            observation_key: observation,
            f"next_{observation_key}": next_observation,
            f"next_{reward_key}": next_reward,
            f"next_{done_key}": next_done,
            f"next_{terminated_key}": next_terminated,
            "action": action,
        }
        td = TensorDict(kwargs, []).unflatten_keys("_")
        loss_val = loss(**kwargs)

        loss_val_td = loss(td)

        torch.testing.assert_close(loss_val_td.get(loss.out_keys[0]), loss_val[0])
        torch.testing.assert_close(loss_val_td.get(loss.out_keys[1]), loss_val[1])

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_dcql_reduction(self, reduction):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        actor = self._create_mock_actor(action_spec_type="one_hot", device=device)
        td = self._create_mock_data_dcql(action_spec_type="one_hot", device=device)
        loss_fn = DiscreteCQLLoss(
            actor, loss_function="l2", delay_value=False, reduction=reduction
        )
        loss_fn.make_value_estimator()
        loss = loss_fn(td)
        if reduction == "none":
            for key in loss.keys():
                if key.startswith("loss"):
                    assert loss[key].shape == td.shape
        else:
            for key in loss.keys():
                if not key.startswith("loss"):
                    continue
                assert loss[key].shape == torch.Size([])


@pytest.mark.skipif(not _has_transformers, reason="requires transformers lib")
class TestPPO(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        action_key=None,
        observation_key="observation",
        sample_log_prob_key=None,
        composite_action_dist=False,
    ):
        # Actor
        action_spec = Bounded(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        net = nn.Sequential(nn.Linear(obs_dim, 2 * action_dim), NormalParamExtractor())
        if composite_action_dist:
            if action_key is None:
                action_key = ("action", "action1")
            else:
                action_key = (action_key, "action1")
            action_spec = Composite({action_key: {"action1": action_spec}})
            distribution_class = functools.partial(
                CompositeDistribution,
                distribution_map={
                    "action1": TanhNormal,
                },
                name_map={
                    "action1": action_key,
                },
                log_prob_key=sample_log_prob_key,
            )
            module_out_keys = [
                ("params", "action1", "loc"),
                ("params", "action1", "scale"),
            ]
            actor_in_keys = ["params"]
        else:
            if action_key is None:
                action_key = "action"
            distribution_class = TanhNormal
            module_out_keys = actor_in_keys = ["loc", "scale"]
        module = TensorDictModule(
            net, in_keys=[observation_key], out_keys=module_out_keys
        )
        actor = ProbabilisticActor(
            module=module,
            distribution_class=distribution_class,
            in_keys=actor_in_keys,
            out_keys=[action_key],
            spec=action_spec,
            return_log_prob=True,
            log_prob_key=sample_log_prob_key,
        )
        return actor.to(device)

    def _create_mock_value(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        out_keys=None,
        observation_key="observation",
    ):
        module = nn.Linear(obs_dim, 1)
        value = ValueOperator(
            module=module,
            in_keys=[observation_key],
            out_keys=out_keys,
        )
        return value.to(device)

    def _create_mock_actor_value(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        composite_action_dist=False,
        sample_log_prob_key="action_log_prob",
    ):
        # Actor
        action_spec = Bounded(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        if composite_action_dist:
            action_spec = Composite({"action": {"action1": action_spec}})
        base_layer = nn.Linear(obs_dim, 5)
        net = nn.Sequential(
            base_layer, nn.Linear(5, 2 * action_dim), NormalParamExtractor()
        )
        if composite_action_dist:
            distribution_class = functools.partial(
                CompositeDistribution,
                distribution_map={
                    "action1": TanhNormal,
                },
                name_map={
                    "action1": ("action", "action1"),
                },
                log_prob_key=sample_log_prob_key,
            )
            module_out_keys = [
                ("params", "action1", "loc"),
                ("params", "action1", "scale"),
            ]
            actor_in_keys = ["params"]
        else:
            distribution_class = TanhNormal
            module_out_keys = actor_in_keys = ["loc", "scale"]
        module = TensorDictModule(
            net, in_keys=["observation"], out_keys=module_out_keys
        )
        actor = ProbabilisticActor(
            module=module,
            distribution_class=distribution_class,
            in_keys=actor_in_keys,
            spec=action_spec,
            return_log_prob=True,
        )
        module = nn.Sequential(base_layer, nn.Linear(5, 1))
        value = ValueOperator(
            module=module,
            in_keys=["observation"],
        )
        return actor.to(device), value.to(device)

    def _create_mock_actor_value_shared(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        composite_action_dist=False,
        sample_log_prob_key="action_log_prob",
    ):
        # Actor
        action_spec = Bounded(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        if composite_action_dist:
            action_spec = Composite({"action": {"action1": action_spec}})
        base_layer = nn.Linear(obs_dim, 5)
        common = TensorDictModule(
            base_layer, in_keys=["observation"], out_keys=["hidden"]
        )
        net = nn.Sequential(nn.Linear(5, 2 * action_dim), NormalParamExtractor())
        if composite_action_dist:
            distribution_class = functools.partial(
                CompositeDistribution,
                distribution_map={
                    "action1": TanhNormal,
                },
                name_map={
                    "action1": ("action", "action1"),
                },
                log_prob_key=sample_log_prob_key,
            )
            module_out_keys = [
                ("params", "action1", "loc"),
                ("params", "action1", "scale"),
            ]
            actor_in_keys = ["params"]
        else:
            distribution_class = TanhNormal
            module_out_keys = actor_in_keys = ["loc", "scale"]
        module = TensorDictModule(net, in_keys=["hidden"], out_keys=module_out_keys)
        actor_head = ProbabilisticActor(
            module=module,
            distribution_class=distribution_class,
            in_keys=actor_in_keys,
            spec=action_spec,
            return_log_prob=True,
        )
        module = nn.Linear(5, 1)
        value_head = ValueOperator(
            module=module,
            in_keys=["hidden"],
        )
        model = ActorValueOperator(common, actor_head, value_head).to(device)
        return model, model.get_policy_operator(), model.get_value_operator()

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=0, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_ppo(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
        observation_key="observation",
        action_key="action",
        reward_key="reward",
        done_key="done",
        terminated_key="terminated",
        sample_log_prob_key="action_log_prob",
        composite_action_dist=False,
    ):
        # create a tensordict
        obs = torch.randn(batch, obs_dim, device=device)
        next_obs = torch.randn(batch, obs_dim, device=device)
        if atoms:
            raise NotImplementedError
        else:
            action = torch.randn(batch, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, 1, device=device)
        done = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        loc_key = "loc"
        scale_key = "scale"
        loc = torch.randn(batch, 4, device=device)
        scale = torch.rand(batch, 4, device=device)
        td = TensorDict(
            batch_size=(batch,),
            source={
                observation_key: obs,
                "next": {
                    observation_key: next_obs,
                    done_key: done,
                    terminated_key: terminated,
                    reward_key: reward,
                },
                action_key: {"action1": action} if composite_action_dist else action,
                sample_log_prob_key: torch.randn_like(action[..., 1]) / 10,
            },
            device=device,
        )
        if composite_action_dist:
            td[("params", "action1", loc_key)] = loc
            td[("params", "action1", scale_key)] = scale
        else:
            td[loc_key] = loc
            td[scale_key] = scale
        return td

    def _create_seq_mock_data_ppo(
        self,
        batch=2,
        T=4,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
        sample_log_prob_key=None,
        action_key=None,
        composite_action_dist=False,
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim, device=device)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action = torch.randn(batch, T, atoms, action_dim, device=device).clamp(
                -1, 1
            )
        else:
            action = torch.randn(batch, T, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = torch.ones(batch, T, dtype=torch.bool, device=device)
        action = action.masked_fill_(~mask.unsqueeze(-1), 0.0)
        params_mean = torch.randn_like(action) / 10
        params_scale = torch.rand_like(action) / 10
        loc = params_mean.masked_fill_(~mask.unsqueeze(-1), 0.0)
        scale = params_scale.masked_fill_(~mask.unsqueeze(-1), 0.0)
        if sample_log_prob_key is None:
            if composite_action_dist:
                sample_log_prob_key = ("action", "action1_log_prob")
            else:
                # conforming to composite_lp_aggregate(False)
                sample_log_prob_key = "action_log_prob"

        if action_key is None:
            if composite_action_dist:
                action_key = ("action", "action1")
            else:
                action_key = "action"
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "next": {
                    "observation": next_obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                    "done": done,
                    "terminated": terminated,
                    "reward": reward.masked_fill_(~mask.unsqueeze(-1), 0.0),
                },
                "collector": {"mask": mask},
                action_key: action,
                sample_log_prob_key: (
                    torch.randn_like(action[..., 1]) / 10
                ).masked_fill_(~mask, 0.0),
            },
            device=device,
            names=[None, "time"],
        )
        if composite_action_dist:
            td[("params", "action1", "loc")] = loc
            td[("params", "action1", "scale")] = scale
        else:
            td["loc"] = loc
            td["scale"] = scale

        return td

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    def test_reset_parameters_recursive(self, loss_class):
        actor = self._create_mock_actor()
        value = self._create_mock_value()
        loss_fn = loss_class(actor, value)
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize("gradient_mode", (True, False))
    @pytest.mark.parametrize("advantage", ("gae", "vtrace", "td", "td_lambda", None))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    @pytest.mark.parametrize("functional", [True, False])
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_ppo(
        self,
        loss_class,
        device,
        gradient_mode,
        advantage,
        td_est,
        functional,
        composite_action_dist,
    ):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_ppo(
            device=device, composite_action_dist=composite_action_dist
        )

        actor = self._create_mock_actor(
            device=device, composite_action_dist=composite_action_dist
        )
        value = self._create_mock_value(device=device)
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage == "vtrace":
            advantage = VTrace(
                gamma=0.9,
                value_network=value,
                actor_network=actor,
                differentiable=gradient_mode,
            )
        elif advantage == "td":
            advantage = TD1Estimator(
                gamma=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage == "td_lambda":
            advantage = TDLambdaEstimator(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage is None:
            pass
        else:
            raise NotImplementedError

        loss_fn = loss_class(
            actor,
            value,
            loss_critic_type="l2",
            functional=functional,
            device=device,
        )
        if composite_action_dist:
            loss_fn.set_keys(
                action=("action", "action1"),
                sample_log_prob=[("action", "action1_log_prob")],
            )
            if advantage is not None:
                advantage.set_keys(sample_log_prob=[("action", "action1_log_prob")])
        if advantage is not None:
            assert not composite_lp_aggregate()
            advantage(td)
        else:
            if td_est is not None:
                loss_fn.make_value_estimator(td_est)

        loss = loss_fn(td)
        if isinstance(loss_fn, KLPENPPOLoss):
            if composite_action_dist:
                kl = loss.pop("kl_approx")
            else:
                kl = loss.pop("kl")
            assert (kl != 0).any()

        loss_critic = loss["loss_critic"]
        loss_objective = loss["loss_objective"] + loss.get("loss_entropy", 0.0)
        loss_critic.backward(retain_graph=True)
        # check that grads are independent and non null
        named_parameters = loss_fn.named_parameters()
        counter = 0
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                counter += 1
                assert "actor" not in name
                assert "critic" in name
            if p.grad is None:
                assert ("actor" in name) or ("target_" in name)
                assert ("critic" not in name) or ("target_" in name)
        assert counter == 2

        value.zero_grad()
        loss_objective.backward()
        counter = 0
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                counter += 1
                assert "actor" in name
                assert "critic" not in name
            if p.grad is None:
                assert ("actor" not in name) or ("target_" in name)
                assert ("critic" in name) or ("target_" in name)
        assert counter == 2
        actor.zero_grad()

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize("gradient_mode", (True, False))
    @pytest.mark.parametrize("advantage", ("gae", "vtrace", "td", "td_lambda", None))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    @pytest.mark.parametrize("functional", [True, False])
    def test_ppo_composite_no_aggregate(
        self, loss_class, device, gradient_mode, advantage, td_est, functional
    ):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_ppo(device=device, composite_action_dist=True)

        actor = self._create_mock_actor(
            device=device,
            composite_action_dist=True,
        )
        value = self._create_mock_value(device=device)
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage == "vtrace":
            advantage = VTrace(
                gamma=0.9,
                value_network=value,
                actor_network=actor,
                differentiable=gradient_mode,
            )
        elif advantage == "td":
            advantage = TD1Estimator(
                gamma=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage == "td_lambda":
            advantage = TDLambdaEstimator(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage is None:
            pass
        else:
            raise NotImplementedError

        loss_fn = loss_class(
            actor,
            value,
            loss_critic_type="l2",
            functional=functional,
            device=device,
        )
        loss_fn.set_keys(
            action=("action", "action1"),
            sample_log_prob=[("action", "action1_log_prob")],
        )
        if advantage is not None:
            advantage.set_keys(sample_log_prob=[("action", "action1_log_prob")])
            advantage(td)
        else:
            if td_est is not None:
                loss_fn.make_value_estimator(td_est)

        loss = loss_fn(td)
        if isinstance(loss_fn, KLPENPPOLoss):
            kl = loss.pop("kl_approx")
            assert (kl != 0).any()
        loss_critic = loss["loss_critic"]
        loss_objective = loss["loss_objective"] + loss.get("loss_entropy", 0.0)
        loss_critic.backward(retain_graph=True)
        # check that grads are independent and non null
        named_parameters = loss_fn.named_parameters()
        counter = 0
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                counter += 1
                assert "actor" not in name
                assert "critic" in name
            if p.grad is None:
                assert ("actor" in name) or ("target_" in name)
                assert ("critic" not in name) or ("target_" in name)
        assert counter == 2

        value.zero_grad()
        loss_objective.backward()
        counter = 0
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                counter += 1
                assert "actor" in name
                assert "critic" not in name
            if p.grad is None:
                assert ("actor" not in name) or ("target_" in name)
                assert ("critic" in name) or ("target_" in name)
        assert counter == 2
        actor.zero_grad()

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize("gradient_mode", (True,))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_ppo_state_dict(
        self, loss_class, device, gradient_mode, composite_action_dist
    ):
        torch.manual_seed(self.seed)

        actor = self._create_mock_actor(
            device=device, composite_action_dist=composite_action_dist
        )
        value = self._create_mock_value(device=device)
        loss_fn = loss_class(
            actor,
            value,
            loss_critic_type="l2",
            device=device,
        )
        sd = loss_fn.state_dict()
        loss_fn2 = loss_class(
            actor,
            value,
            loss_critic_type="l2",
            device=device,
        )
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize("advantage", ("gae", "vtrace", "td", "td_lambda", None))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_ppo_shared(self, loss_class, device, advantage, composite_action_dist):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_ppo(
            device=device, composite_action_dist=composite_action_dist
        )

        actor, value = self._create_mock_actor_value(
            device=device, composite_action_dist=composite_action_dist
        )
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9,
                lmbda=0.9,
                value_network=value,
            )
        elif advantage == "vtrace":
            advantage = VTrace(
                gamma=0.9,
                value_network=value,
                actor_network=actor,
            )
        elif advantage == "td":
            advantage = TD1Estimator(
                gamma=0.9,
                value_network=value,
            )
        elif advantage == "td_lambda":
            advantage = TDLambdaEstimator(
                gamma=0.9,
                lmbda=0.9,
                value_network=value,
            )
        elif advantage is None:
            pass
        else:
            raise NotImplementedError
        loss_fn = loss_class(
            actor,
            value,
            loss_critic_type="l2",
            separate_losses=True,
            device=device,
        )

        if advantage is not None:
            if composite_action_dist:
                advantage.set_keys(sample_log_prob=[("action", "action1_log_prob")])
            advantage(td)

        if composite_action_dist:
            loss_fn.set_keys(
                action=("action", "action1"),
                sample_log_prob=[("action", "action1_log_prob")],
            )
        loss = loss_fn(td)

        loss_critic = loss["loss_critic"]
        loss_objective = loss["loss_objective"] + loss.get("loss_entropy", 0.0)
        loss_critic.backward(retain_graph=True)
        # check that grads are independent and non null
        named_parameters = loss_fn.named_parameters()
        counter = 0
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                counter += 1
                assert "actor" not in name
                assert "critic" in name
            if p.grad is None:
                assert ("actor" in name) or ("target_" in name)
                assert ("critic" not in name) or ("target_" in name)
        assert counter == 2

        value.zero_grad()
        loss_objective.backward()
        named_parameters = loss_fn.named_parameters()
        counter = 0
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                counter += 1
                assert "actor" in name
                assert "critic" not in name
            if p.grad is None:
                assert ("actor" not in name) or ("target_" in name)
                assert ("critic" in name) or ("target_" in name)
        actor.zero_grad()
        assert counter == 4

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize(
        "advantage",
        (
            "gae",
            "vtrace",
            "td",
            "td_lambda",
        ),
    )
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("separate_losses", [True, False])
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_ppo_shared_seq(
        self,
        loss_class,
        device,
        advantage,
        separate_losses,
        composite_action_dist,
    ):
        """Tests PPO with shared module with and without passing twice across the common module."""
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_ppo(
            device=device, composite_action_dist=composite_action_dist
        )

        model, actor, value = self._create_mock_actor_value_shared(
            device=device, composite_action_dist=composite_action_dist
        )
        value2 = value[-1]  # prune the common module
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9,
                lmbda=0.9,
                value_network=value,
            )
        elif advantage == "vtrace":
            advantage = VTrace(
                gamma=0.9,
                value_network=value,
                actor_network=actor,
            )
        elif advantage == "td":
            advantage = TD1Estimator(
                gamma=0.9,
                value_network=value,
            )
        elif advantage == "td_lambda":
            advantage = TDLambdaEstimator(
                gamma=0.9,
                lmbda=0.9,
                value_network=value,
            )
        else:
            raise NotImplementedError
        loss_fn = loss_class(
            actor,
            value,
            loss_critic_type="l2",
            separate_losses=separate_losses,
            entropy_coeff=0.0,
            device=device,
        )

        loss_fn2 = loss_class(
            actor,
            value2,
            loss_critic_type="l2",
            separate_losses=separate_losses,
            entropy_coeff=0.0,
            device=device,
        )

        if advantage is not None:
            if composite_action_dist:
                advantage.set_keys(sample_log_prob=[("action", "action1_log_prob")])
            advantage(td)

        if composite_action_dist:
            loss_fn.set_keys(
                action=("action", "action1"),
                sample_log_prob=[("action", "action1_log_prob")],
            )
            loss_fn2.set_keys(
                action=("action", "action1"),
                sample_log_prob=[("action", "action1_log_prob")],
            )

        loss = loss_fn(td).exclude("entropy")
        if composite_action_dist:
            loss = loss.exclude("composite_entropy")

        sum(val for key, val in loss.items() if key.startswith("loss_")).backward()
        grad = TensorDict(dict(model.named_parameters()), []).apply(
            lambda x: x.grad.clone()
        )
        loss2 = loss_fn2(td).exclude("entropy")
        if composite_action_dist:
            loss2 = loss2.exclude("composite_entropy")

        model.zero_grad()
        sum(val for key, val in loss2.items() if key.startswith("loss_")).backward()
        grad2 = TensorDict(dict(model.named_parameters()), []).apply(
            lambda x: x.grad.clone()
        )
        if composite_action_dist and loss_class is KLPENPPOLoss:
            # KL computation for composite dist is based on randomly
            # sampled data, thus will not be the same.
            # Similarly, objective loss depends on the KL, so ir will
            # not be the same either.
            # Finally, gradients will be different too.
            loss.pop("kl", None)
            loss2.pop("kl", None)
            loss.pop("loss_objective", None)
            loss2.pop("loss_objective", None)
            assert_allclose_td(loss, loss2)
        else:
            assert_allclose_td(loss, loss2)
            assert_allclose_td(grad, grad2)
        model.zero_grad()

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found, {FUNCTORCH_ERR}"
    )
    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize("gradient_mode", (True, False))
    @pytest.mark.parametrize("advantage", ("gae", "vtrace", "td", "td_lambda", None))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_ppo_diff(
        self, loss_class, device, gradient_mode, advantage, composite_action_dist
    ):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_ppo(
            device=device, composite_action_dist=composite_action_dist
        )

        actor = self._create_mock_actor(
            device=device, composite_action_dist=composite_action_dist
        )
        value = self._create_mock_value(device=device)
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage == "vtrace":
            advantage = VTrace(
                gamma=0.9,
                value_network=value,
                actor_network=actor,
                differentiable=gradient_mode,
            )
        elif advantage == "td":
            advantage = TD1Estimator(
                gamma=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage == "td_lambda":
            advantage = TDLambdaEstimator(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage is None:
            pass
        else:
            raise NotImplementedError

        loss_fn = loss_class(
            actor,
            value,
            loss_critic_type="l2",
            device=device,
        )

        params = TensorDict.from_module(loss_fn, as_module=True)

        # fill params with zero
        def zero_param(p):
            if isinstance(p, nn.Parameter):
                p.data.zero_()

        params.apply(zero_param, filter_empty=True)

        # assert len(list(floss_fn.parameters())) == 0
        with params.to_module(loss_fn):
            if advantage is not None:
                if composite_action_dist:
                    advantage.set_keys(sample_log_prob=[("action", "action1_log_prob")])
                advantage(td)
            if composite_action_dist:
                loss_fn.set_keys(
                    action=("action", "action1"),
                    sample_log_prob=[("action", "action1_log_prob")],
                )
            loss = loss_fn(td)

        loss_critic = loss["loss_critic"]
        loss_objective = loss["loss_objective"] + loss.get("loss_entropy", 0.0)
        loss_critic.backward(retain_graph=True)
        # check that grads are independent and non null
        named_parameters = loss_fn.named_parameters()
        for name, p in params.items(True, True):
            if isinstance(name, tuple):
                name = "-".join(name)
            if not isinstance(p, nn.Parameter):
                continue
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" not in name
                assert "critic" in name
            if p.grad is None:
                assert "actor" in name
                assert "critic" not in name

        for p in params.values(True, True):
            p.grad = None
        loss_objective.backward()
        named_parameters = loss_fn.named_parameters()
        for name, other_p in named_parameters:
            p = params.get(tuple(name.split(".")))
            assert other_p.shape == p.shape
            assert other_p.dtype == p.dtype
            assert other_p.device == p.device
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" in name
                assert "critic" not in name
            if p.grad is None:
                assert "actor" not in name
                assert "critic" in name
        for param in params.values(True, True):
            param.grad = None

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize(
        "td_est",
        [
            ValueEstimators.TD1,
            ValueEstimators.TD0,
            ValueEstimators.GAE,
            ValueEstimators.VTrace,
            ValueEstimators.TDLambda,
        ],
    )
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_ppo_tensordict_keys(self, loss_class, td_est, composite_action_dist):
        assert not composite_lp_aggregate()
        actor = self._create_mock_actor(composite_action_dist=composite_action_dist)
        value = self._create_mock_value()

        loss_fn = loss_class(actor, value, loss_critic_type="l2")

        default_keys = {
            "advantage": "advantage",
            "value_target": "value_target",
            "value": "state_value",
            "sample_log_prob": "action_log_prob"
            if not composite_action_dist
            else ("action", "action1_log_prob"),
            "action": "action" if not composite_action_dist else ("action", "action1"),
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
        }

        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
            td_est=td_est,
        )

        value_key = "state_value_test"
        value = self._create_mock_value(out_keys=[value_key])
        loss_fn = loss_class(actor, value, loss_critic_type="l2")

        key_mapping = {
            "advantage": ("advantage", "advantage_new"),
            "value_target": ("value_target", "value_target_new"),
            "value": ("value", value_key),
            "reward": ("reward", "reward_test"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize("advantage", ("gae", "vtrace", "td", "td_lambda", None))
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_ppo_tensordict_keys_run(self, loss_class, advantage, td_est):
        """Test PPO loss module with non-default tensordict keys."""
        torch.manual_seed(self.seed)
        gradient_mode = True
        tensor_keys = {
            "advantage": "advantage_test",
            "value_target": "value_target_test",
            "value": "state_value_test",
            "sample_log_prob": "action_log_prob_test",
            "action": "action_test",
        }

        td = self._create_seq_mock_data_ppo(
            sample_log_prob_key=tensor_keys["sample_log_prob"],
            action_key=tensor_keys["action"],
        )
        actor = self._create_mock_actor(
            sample_log_prob_key=tensor_keys["sample_log_prob"],
            action_key=tensor_keys["action"],
        )
        value = self._create_mock_value(out_keys=[tensor_keys["value"]])

        if advantage == "gae":
            advantage = GAE(
                gamma=0.9,
                lmbda=0.9,
                value_network=value,
                differentiable=gradient_mode,
            )
        elif advantage == "vtrace":
            advantage = VTrace(
                gamma=0.9,
                value_network=value,
                actor_network=actor,
                differentiable=gradient_mode,
            )
        elif advantage == "td":
            advantage = TD1Estimator(
                gamma=0.9,
                value_network=value,
                differentiable=gradient_mode,
            )
        elif advantage == "td_lambda":
            advantage = TDLambdaEstimator(
                gamma=0.9,
                lmbda=0.9,
                value_network=value,
                differentiable=gradient_mode,
            )
        elif advantage is None:
            pass
        else:
            raise NotImplementedError

        loss_fn = loss_class(actor, value, loss_critic_type="l2")
        loss_fn.set_keys(**tensor_keys)
        if advantage is not None:
            # collect tensordict key names for the advantage module
            adv_keys = {
                key: value
                for key, value in tensor_keys.items()
                if key in asdict(GAE._AcceptedKeys()).keys()
            }
            advantage.set_keys(**adv_keys)
            advantage(td)
        else:
            if td_est is not None:
                loss_fn.make_value_estimator(td_est)

        loss = loss_fn(td)

        loss_critic = loss["loss_critic"]
        loss_objective = loss["loss_objective"] + loss.get("loss_entropy", 0.0)
        loss_critic.backward(retain_graph=True)
        # check that grads are independent and non null
        named_parameters = loss_fn.named_parameters()
        counter = 0
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                counter += 1
                assert "actor" not in name
                assert "critic" in name
            if p.grad is None:
                assert ("actor" in name) or ("target" in name)
                assert ("critic" not in name) or ("target" in name)
        assert counter == 2

        value.zero_grad()
        loss_objective.backward()
        counter = 0
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                counter += 1
                assert "actor" in name
                assert "critic" not in name
            if p.grad is None:
                assert ("actor" not in name) or ("target" in name)
                assert ("critic" in name) or ("target" in name)
        assert counter == 2
        actor.zero_grad()

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize("action_key", ["action", "action2"])
    @pytest.mark.parametrize("sample_log_prob_key", ["samplelogprob", "samplelogprob2"])
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    @pytest.mark.parametrize(
        "composite_action_dist",
        [False],
    )
    def test_ppo_notensordict(
        self,
        loss_class,
        action_key,
        sample_log_prob_key,
        observation_key,
        reward_key,
        done_key,
        terminated_key,
        composite_action_dist,
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_ppo(
            observation_key=observation_key,
            action_key=action_key,
            sample_log_prob_key=sample_log_prob_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
            composite_action_dist=composite_action_dist,
        )

        actor = self._create_mock_actor(
            observation_key=observation_key,
            sample_log_prob_key=sample_log_prob_key,
            composite_action_dist=composite_action_dist,
            action_key=action_key,
        )
        value = self._create_mock_value(observation_key=observation_key)

        loss = loss_class(actor_network=actor, critic_network=value)
        loss.set_keys(
            action=action_key,
            reward=reward_key,
            done=done_key,
            terminated=terminated_key,
            sample_log_prob=sample_log_prob_key,
        )

        kwargs = {
            action_key: td.get(action_key),
            observation_key: td.get(observation_key),
            sample_log_prob_key: td.get(sample_log_prob_key),
            f"next_{reward_key}": td.get(("next", reward_key)),
            f"next_{done_key}": td.get(("next", done_key)),
            f"next_{terminated_key}": td.get(("next", terminated_key)),
            f"next_{observation_key}": td.get(("next", observation_key)),
        }
        if loss_class is KLPENPPOLoss:
            loc_key = "params" if composite_action_dist else "loc"
            scale_key = "params" if composite_action_dist else "scale"
            kwargs.update({loc_key: td.get(loc_key), scale_key: td.get(scale_key)})

        td = TensorDict(kwargs, td.batch_size, names=["time"]).unflatten_keys("_")

        # setting the seed for each loss so that drawing the random samples from
        # value network leads to same numbers for both runs
        torch.manual_seed(self.seed)
        beta = getattr(loss, "beta", None)
        if beta is not None:
            beta = beta.clone()
        loss_val = loss(**kwargs)
        torch.manual_seed(self.seed)
        if beta is not None:

            loss.beta = beta.clone()
        loss_val_td = loss(td)

        for i, out_key in enumerate(loss.out_keys):
            torch.testing.assert_close(
                loss_val_td.get(out_key), loss_val[i], msg=out_key
            )

        # test select
        torch.manual_seed(self.seed)
        if beta is not None:
            loss.beta = beta.clone()
        loss.select_out_keys("loss_objective", "loss_critic")
        if torch.__version__ >= "2.0.0":
            loss_obj, loss_crit = loss(**kwargs)
        else:
            with pytest.raises(
                RuntimeError,
                match="You are likely using tensordict.nn.dispatch with keyword arguments",
            ):
                loss_obj, loss_crit = loss(**kwargs)
            return
        assert loss_obj == loss_val_td.get("loss_objective")
        assert loss_crit == loss_val_td.get("loss_critic")

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_ppo_reduction(self, reduction, loss_class, composite_action_dist):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        td = self._create_seq_mock_data_ppo(
            device=device, composite_action_dist=composite_action_dist
        )
        actor = self._create_mock_actor(
            device=device, composite_action_dist=composite_action_dist
        )
        value = self._create_mock_value(device=device)
        advantage = GAE(
            gamma=0.9,
            lmbda=0.9,
            value_network=value,
        )
        loss_fn = loss_class(
            actor,
            value,
            loss_critic_type="l2",
            reduction=reduction,
        )
        advantage(td)
        if composite_action_dist:
            loss_fn.set_keys(
                action=("action", "action1"),
                sample_log_prob=[("action", "action1_log_prob")],
            )
        loss = loss_fn(td)
        if reduction == "none":
            for key in loss.keys():
                if key.startswith("loss_"):
                    assert loss[key].shape == td.shape, key
        else:
            for key in loss.keys():
                if not key.startswith("loss_"):
                    continue
                assert loss[key].shape == torch.Size([])

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize("clip_value", [True, False, None, 0.5, torch.tensor(0.5)])
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_ppo_value_clipping(
        self, clip_value, loss_class, device, composite_action_dist
    ):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_ppo(
            device=device, composite_action_dist=composite_action_dist
        )
        actor = self._create_mock_actor(
            device=device, composite_action_dist=composite_action_dist
        )
        value = self._create_mock_value(device=device)
        advantage = GAE(
            gamma=0.9,
            lmbda=0.9,
            value_network=value,
        )

        if isinstance(clip_value, bool) and loss_class is not ClipPPOLoss:
            with pytest.raises(
                ValueError,
                match=f"clip_value must be a float or a scalar tensor, got {clip_value}.",
            ):
                loss_fn = loss_class(
                    actor,
                    value,
                    loss_critic_type="l2",
                    clip_value=clip_value,
                    device=device,
                )

        else:
            loss_fn = loss_class(
                actor,
                value,
                loss_critic_type="l2",
                clip_value=clip_value,
                device=device,
            )
            advantage(td)
            if composite_action_dist:
                loss_fn.set_keys(
                    action=("action", "action1"),
                    sample_log_prob=[("action", "action1_log_prob")],
                )

            value = td.pop(loss_fn.tensor_keys.value)

            if clip_value:
                # Test it fails without value key
                with pytest.raises(
                    KeyError,
                    match=f"clip_value is set to {loss_fn.clip_value}, but the key "
                    "state_value was not found in the input tensordict. "
                    "Make sure that the.*passed to PPO exists in "
                    "the input tensordict.",
                ):
                    loss = loss_fn(td)

            # Add value back to td
            td.set(loss_fn.tensor_keys.value, value)

            # Test it works with value key
            loss = loss_fn(td)
            assert "loss_critic" in loss.keys()

    def test_ppo_composite_dists(self):
        d = torch.distributions

        make_params = TensorDictModule(
            lambda: (
                torch.ones(4),
                torch.ones(4),
                torch.ones(4),
                torch.ones(4),
                torch.ones(4),
                torch.ones(4),
                torch.ones(4, 2),
                torch.ones(4, 2),
                torch.ones(4, 10) / 10,
                torch.zeros(4, 10),
                torch.ones(4, 10),
            ),
            in_keys=[],
            out_keys=[
                ("params", "gamma1", "concentration"),
                ("params", "gamma1", "rate"),
                ("params", "gamma2", "concentration"),
                ("params", "gamma2", "rate"),
                ("params", "gamma3", "concentration"),
                ("params", "gamma3", "rate"),
                ("params", "Kumaraswamy", "concentration0"),
                ("params", "Kumaraswamy", "concentration1"),
                ("params", "mixture", "logits"),
                ("params", "mixture", "loc"),
                ("params", "mixture", "scale"),
            ],
        )

        def mixture_constructor(logits, loc, scale):
            return d.MixtureSameFamily(
                d.Categorical(logits=logits), d.Normal(loc=loc, scale=scale)
            )

        dist_constructor = functools.partial(
            CompositeDistribution,
            distribution_map={
                "gamma1": d.Gamma,
                "gamma2": d.Gamma,
                "gamma3": d.Gamma,
                "Kumaraswamy": d.Kumaraswamy,
                "mixture": mixture_constructor,
            },
            name_map={
                "gamma1": ("agent0", "action", "action1", "sub_action1"),
                "gamma2": ("agent0", "action", "action1", "sub_action2"),
                "gamma3": ("agent0", "action", "action2"),
                "Kumaraswamy": ("agent1", "action"),
                "mixture": ("agent2"),
            },
        )
        policy = ProbSeq(
            make_params,
            ProbabilisticTensorDictModule(
                in_keys=["params"],
                out_keys=[
                    ("agent0", "action", "action1", "sub_action1"),
                    ("agent0", "action", "action1", "sub_action2"),
                    ("agent0", "action", "action2"),
                    ("agent1", "action"),
                    ("agent2"),
                ],
                distribution_class=dist_constructor,
                return_log_prob=True,
                default_interaction_type=InteractionType.RANDOM,
            ),
        )
        # We want to make sure there is no warning
        td = policy(TensorDict(batch_size=[4]))
        assert isinstance(
            policy.get_dist(td).log_prob(td),
            TensorDict,
        )
        assert isinstance(
            policy.log_prob(td),
            TensorDict,
        )
        value_operator = Seq(
            WrapModule(
                lambda td: td.set("state_value", torch.ones((*td.shape, 1))),
                out_keys=["state_value"],
            )
        )
        for cls in (PPOLoss, ClipPPOLoss, KLPENPPOLoss):
            data = policy(TensorDict(batch_size=[4]))
            data.set(
                "next",
                TensorDict(
                    reward=torch.randn(4, 1), done=torch.zeros(4, 1, dtype=torch.bool)
                ),
            )
            scalar_entropy = 0.07
            ppo = cls(policy, value_operator, entropy_coeff=scalar_entropy)
            ppo.set_keys(
                action=[
                    ("agent0", "action", "action1", "sub_action1"),
                    ("agent0", "action", "action1", "sub_action2"),
                    ("agent0", "action", "action2"),
                    ("agent1", "action"),
                    ("agent2"),
                ],
                sample_log_prob=[
                    ("agent0", "action", "action1", "sub_action1_log_prob"),
                    ("agent0", "action", "action1", "sub_action2_log_prob"),
                    ("agent0", "action", "action2_log_prob"),
                    ("agent1", "action_log_prob"),
                    ("agent2_log_prob"),
                ],
            )
            loss = ppo(data)
            composite_entropy = loss["composite_entropy"]
            entropy = _sum_td_features(composite_entropy)
            expected_loss = -(scalar_entropy * entropy).mean()  # batch mean
            torch.testing.assert_close(
                loss["loss_entropy"], expected_loss, rtol=1e-5, atol=1e-7
            )
            loss.sum(reduce=True)

            # keep per-head entropies instead of the aggregated tensor
            set_composite_lp_aggregate(False).set()
            coef_map = {
                ("agent0", "action", "action1", "sub_action1_log_prob"): 0.02,
                ("agent0", "action", "action1", "sub_action2_log_prob"): 0.01,
                ("agent0", "action", "action2_log_prob"): 0.01,
                ("agent1", "action_log_prob"): 0.01,
                "agent2_log_prob": 0.01,
            }
            ppo_weighted = cls(policy, value_operator, entropy_coeff=coef_map)
            ppo_weighted.set_keys(
                action=[
                    ("agent0", "action", "action1", "sub_action1"),
                    ("agent0", "action", "action1", "sub_action2"),
                    ("agent0", "action", "action2"),
                    ("agent1", "action"),
                    ("agent2"),
                ],
                sample_log_prob=[
                    ("agent0", "action", "action1", "sub_action1_log_prob"),
                    ("agent0", "action", "action1", "sub_action2_log_prob"),
                    ("agent0", "action", "action2_log_prob"),
                    ("agent1", "action_log_prob"),
                    ("agent2_log_prob"),
                ],
            )
            loss = ppo_weighted(data)
            composite_entropy = loss["composite_entropy"]

            # sanity check: loss_entropy is scalar + finite
            assert loss["loss_entropy"].ndim == 0
            assert torch.isfinite(loss["loss_entropy"])
            # Check individual loss is computed with the right weights
            expected_loss = 0.0
            for i, (_, head_entropy) in enumerate(
                composite_entropy.items(include_nested=True, leaves_only=True)
            ):
                expected_loss -= (
                    coef_map[list(coef_map.keys())[i]] * head_entropy
                ).mean()
            torch.testing.assert_close(
                loss["loss_entropy"], expected_loss, rtol=1e-5, atol=1e-7
            )

    def test_ppo_marl_aggregate(self):
        env = MARLEnv()

        def primer(td):
            params = TensorDict(
                agents=TensorDict(
                    dirich=TensorDict(
                        concentration=env.action_spec["agents", "dirich"].one()
                    ),
                    categ=TensorDict(logits=env.action_spec["agents", "categ"].one()),
                    batch_size=env.action_spec["agents"].shape,
                ),
                batch_size=td.batch_size,
            )
            td.set("params", params)
            return td

        policy = ProbabilisticTensorDictSequential(
            primer,
            env.make_composite_dist(),
            # return_composite=True,
        )
        output = policy(env.fake_tensordict())
        assert output.shape == env.batch_size
        assert (
            output["agents", "dirich_log_prob"].shape == env.batch_size + env.n_agents
        )
        assert output["agents", "categ_log_prob"].shape == env.batch_size + env.n_agents

        output["advantage"] = output["next", "agents", "reward"].clone()
        output["value_target"] = output["next", "agents", "reward"].clone()
        critic = TensorDictModule(
            lambda obs: obs.new_zeros((*obs.shape[:-1], 1)),
            in_keys=list(env.full_observation_spec.keys(True, True)),
            out_keys=["state_value"],
        )
        ppo = ClipPPOLoss(actor_network=policy, critic_network=critic)
        ppo.set_keys(action=list(env.full_action_spec.keys(True, True)))
        assert isinstance(ppo.tensor_keys.action, list)
        ppo(output)

    def _make_entropy_loss(self, entropy_coeff):
        actor, critic = self._create_mock_actor_value()
        return PPOLoss(actor, critic, entropy_coeff=entropy_coeff)

    def test_weighted_entropy_scalar(self):
        loss = self._make_entropy_loss(entropy_coeff=0.5)
        entropy = torch.tensor(2.0)
        out = loss._weighted_loss_entropy(entropy)
        torch.testing.assert_close(out, torch.tensor(-1.0))

    def test_weighted_entropy_mapping(self):
        coef = {("head_0", "action_log_prob"): 0.3, ("head_1", "action_log_prob"): 0.7}
        loss = self._make_entropy_loss(entropy_coeff=coef)
        entropy = TensorDict(
            {
                "head_0": {"action_log_prob": torch.tensor(1.0)},
                "head_1": {"action_log_prob": torch.tensor(2.0)},
            },
            [],
        )
        out = loss._weighted_loss_entropy(entropy)
        expected = -(
            coef[("head_0", "action_log_prob")] * 1.0
            + coef[("head_1", "action_log_prob")] * 2.0
        )
        torch.testing.assert_close(out, torch.tensor(expected))

    def test_weighted_entropy_mapping_missing_key(self):
        loss = self._make_entropy_loss(entropy_coeff={"head_not_present": 0.5})
        entropy = TensorDict({"head_0": {"action_log_prob": torch.tensor(1.0)}}, [])
        with pytest.raises(KeyError):
            loss._weighted_loss_entropy(entropy)

    def test_critic_loss_tensordict(self):
        # Creates a dummy actor.
        actor, _ = self._create_mock_actor_value()

        # Creates a critic that produces a tensordict of values.
        class CompositeValueNetwork(nn.Module):
            def forward(self, _) -> tuple[torch.Tensor, torch.Tensor]:
                return torch.tensor([0.0]), torch.tensor([0.0])

        critic = TensorDictModule(
            CompositeValueNetwork(),
            in_keys=["state"],
            out_keys=[("state_value", "value_0"), ("state_value", "value_1")],
        )

        # Creates the loss and its input tensordict.
        loss = ClipPPOLoss(actor, critic, loss_critic_type="l2", clip_value=0.1)
        td = TensorDict(
            {
                "state": torch.tensor([0.0]),
                "value_target": TensorDict(
                    {"value_0": torch.tensor([-1.0]), "value_1": torch.tensor([2.0])}
                ),
                # Log an existing 'state_value' for the 'clip_fraction'
                "state_value": TensorDict(
                    {"value_0": torch.tensor([0.0]), "value_1": torch.tensor([0.0])}
                ),
            },
            batch_size=(1,),
        )

        critic_loss, clip_fraction, explained_variance = loss.loss_critic(td)

        assert isinstance(critic_loss, TensorDict)
        assert "value_0" in critic_loss.keys() and "value_1" in critic_loss.keys()
        torch.testing.assert_close(critic_loss["value_0"], torch.tensor([1.0]))
        torch.testing.assert_close(critic_loss["value_1"], torch.tensor([4.0]))

        assert isinstance(clip_fraction, TensorDict)
        assert isinstance(explained_variance, TensorDict)


class TestA2C(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        action_key="action",
        observation_key="observation",
        composite_action_dist=False,
        sample_log_prob_key=None,
    ):
        # Actor
        action_spec = Bounded(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        if composite_action_dist:
            action_spec = Composite({action_key: {"action1": action_spec}})
        net = nn.Sequential(nn.Linear(obs_dim, 2 * action_dim), NormalParamExtractor())
        if composite_action_dist:
            distribution_class = functools.partial(
                CompositeDistribution,
                distribution_map={
                    "action1": TanhNormal,
                },
                name_map={
                    "action1": (action_key, "action1"),
                },
            )
            module_out_keys = [
                ("params", "action1", "loc"),
                ("params", "action1", "scale"),
            ]
            actor_in_keys = ["params"]
        else:
            distribution_class = TanhNormal
            module_out_keys = actor_in_keys = ["loc", "scale"]
        module = TensorDictModule(
            net, in_keys=[observation_key], out_keys=module_out_keys
        )
        actor = ProbabilisticActor(
            module=module,
            in_keys=actor_in_keys,
            out_keys=[action_key],
            spec=action_spec,
            distribution_class=distribution_class,
            return_log_prob=True,
            log_prob_key=sample_log_prob_key,
        )
        return actor.to(device)

    def _create_mock_value(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        out_keys=None,
        observation_key="observation",
    ):
        module = nn.Linear(obs_dim, 1)
        value = ValueOperator(
            module=module,
            in_keys=[observation_key],
            out_keys=out_keys,
        )
        return value.to(device)

    def _create_mock_common_layer_setup(
        self,
        n_obs=3,
        n_act=4,
        ncells=4,
        batch=2,
        n_hidden=2,
        T=10,
        composite_action_dist=False,
    ):
        common_net = MLP(
            num_cells=ncells,
            in_features=n_obs,
            depth=3,
            out_features=n_hidden,
        )
        actor_net = MLP(
            num_cells=ncells,
            in_features=n_hidden,
            depth=1,
            out_features=2 * n_act,
        )
        value_net = MLP(
            in_features=n_hidden,
            num_cells=ncells,
            depth=1,
            out_features=1,
        )
        batch = [batch, T]
        action = torch.randn(*batch, n_act)
        td = TensorDict(
            {
                "obs": torch.randn(*batch, n_obs),
                "action": {"action1": action} if composite_action_dist else action,
                "action_log_prob": torch.randn(*batch),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
                "terminated": torch.zeros(*batch, 1, dtype=torch.bool),
                "next": {
                    "obs": torch.randn(*batch, n_obs),
                    "reward": torch.randn(*batch, 1),
                    "done": torch.zeros(*batch, 1, dtype=torch.bool),
                    "terminated": torch.zeros(*batch, 1, dtype=torch.bool),
                },
            },
            batch,
            names=[None, "time"],
        )
        common = Mod(common_net, in_keys=["obs"], out_keys=["hidden"])

        if composite_action_dist:
            distribution_class = functools.partial(
                CompositeDistribution,
                distribution_map={
                    "action1": TanhNormal,
                },
                name_map={
                    "action1": ("action", "action1"),
                },
            )
            module_out_keys = [
                ("params", "action1", "loc"),
                ("params", "action1", "scale"),
            ]
            actor_in_keys = ["params"]
        else:
            distribution_class = TanhNormal
            module_out_keys = actor_in_keys = ["loc", "scale"]

        actor = ProbSeq(
            common,
            Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
            Mod(NormalParamExtractor(), in_keys=["param"], out_keys=module_out_keys),
            ProbMod(
                in_keys=actor_in_keys,
                out_keys=["action"],
                distribution_class=distribution_class,
            ),
        )
        critic = Seq(
            common, Mod(value_net, in_keys=["hidden"], out_keys=["state_value"])
        )
        actor(td.clone())
        critic(td.clone())
        return actor, critic, common, td

    def _create_seq_mock_data_a2c(
        self,
        batch=2,
        T=4,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
        action_key="action",
        observation_key="observation",
        reward_key="reward",
        done_key="done",
        terminated_key="terminated",
        sample_log_prob_key="action_log_prob",
        composite_action_dist=False,
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim, device=device)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action = torch.randn(batch, T, atoms, action_dim, device=device).clamp(
                -1, 1
            )
        else:
            action = torch.randn(batch, T, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = ~torch.zeros(batch, T, dtype=torch.bool, device=device)
        action = action.masked_fill_(~mask.unsqueeze(-1), 0.0)
        params_mean = torch.randn_like(action) / 10
        params_scale = torch.rand_like(action) / 10
        loc = params_mean.masked_fill_(~mask.unsqueeze(-1), 0.0)
        scale = params_scale.masked_fill_(~mask.unsqueeze(-1), 0.0)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                observation_key: obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "next": {
                    observation_key: next_obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                    done_key: done,
                    terminated_key: terminated,
                    reward_key: reward.masked_fill_(~mask.unsqueeze(-1), 0.0),
                },
                "collector": {"mask": mask},
                action_key: {"action1": action} if composite_action_dist else action,
                sample_log_prob_key: torch.randn_like(action[..., 1]).masked_fill_(
                    ~mask, 0.0
                )
                / 10,
            },
            device=device,
            names=[None, "time"],
        )
        if composite_action_dist:
            td[("params", "action1", "loc")] = loc
            td[("params", "action1", "scale")] = scale
        else:
            td["loc"] = loc
            td["scale"] = scale
        return td

    def test_reset_parameters_recursive(self):
        actor = self._create_mock_actor()
        value = self._create_mock_value()
        loss_fn = A2CLoss(actor, value)
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("gradient_mode", (True, False))
    @pytest.mark.parametrize("advantage", ("gae", "vtrace", "td", "td_lambda", None))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    @pytest.mark.parametrize("functional", (True, False))
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_a2c(
        self,
        device,
        gradient_mode,
        advantage,
        td_est,
        functional,
        composite_action_dist,
    ):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_a2c(
            device=device, composite_action_dist=composite_action_dist
        )

        actor = self._create_mock_actor(
            device=device, composite_action_dist=composite_action_dist
        )
        value = self._create_mock_value(device=device)
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage == "vtrace":
            advantage = VTrace(
                gamma=0.9,
                value_network=value,
                actor_network=actor,
                differentiable=gradient_mode,
            )
        elif advantage == "td":
            advantage = TD1Estimator(
                gamma=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage == "td_lambda":
            advantage = TDLambdaEstimator(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage is None:
            pass
        else:
            raise NotImplementedError

        loss_fn = A2CLoss(
            actor,
            value,
            loss_critic_type="l2",
            functional=functional,
        )

        def set_requires_grad(tensor, requires_grad):
            tensor.requires_grad = requires_grad
            return tensor

        # Check error is raised when actions require grads
        if composite_action_dist:
            td["action"].apply_(lambda x: set_requires_grad(x, True))
        else:
            td["action"].requires_grad = True
        with pytest.raises(
            RuntimeError,
            match="tensordict stored action requires grad.",
        ):
            _ = loss_fn._log_probs(td)
        if composite_action_dist:
            td["action"].apply_(lambda x: set_requires_grad(x, False))
        else:
            td["action"].requires_grad = False

        td = td.exclude(loss_fn.tensor_keys.value_target)
        if advantage is not None:
            advantage.set_keys(
                sample_log_prob=actor.log_prob_keys
                if composite_action_dist
                else "action_log_prob"
            )
            advantage(td)
        elif td_est is not None:
            loss_fn.make_value_estimator(td_est)
        loss = loss_fn(td)

        loss_critic = loss["loss_critic"]
        loss_objective = loss["loss_objective"] + loss.get("loss_entropy", 0.0)
        loss_critic.backward(retain_graph=True)
        # check that grads are independent and non null
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" not in name
                assert "critic" in name
            if p.grad is None:
                assert ("actor" in name) or ("target_" in name)
                assert ("critic" not in name) or ("target_" in name)

        value.zero_grad()
        for n, p in loss_fn.named_parameters():
            assert p.grad is None or p.grad.norm() == 0, n
        loss_objective.backward()
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" in name
                assert "critic" not in name
            if p.grad is None:
                assert ("actor" not in name) or ("target_" in name)
                assert ("critic" in name) or ("target_" in name)
        actor.zero_grad()

        # test reset
        loss_fn.reset()

    @pytest.mark.parametrize("gradient_mode", (True, False))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_a2c_state_dict(self, device, gradient_mode, composite_action_dist):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            device=device, composite_action_dist=composite_action_dist
        )
        value = self._create_mock_value(device=device)
        loss_fn = A2CLoss(actor, value, loss_critic_type="l2")
        sd = loss_fn.state_dict()
        loss_fn2 = A2CLoss(actor, value, loss_critic_type="l2")
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("separate_losses", [False, True])
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_a2c_separate_losses(self, separate_losses, composite_action_dist):
        torch.manual_seed(self.seed)
        actor, critic, common, td = self._create_mock_common_layer_setup(
            composite_action_dist=composite_action_dist
        )
        loss_fn = A2CLoss(
            actor_network=actor,
            critic_network=critic,
            separate_losses=separate_losses,
        )

        def set_requires_grad(tensor, requires_grad):
            tensor.requires_grad = requires_grad
            return tensor

        # Check error is raised when actions require grads
        if composite_action_dist:
            td["action"].apply_(lambda x: set_requires_grad(x, True))
        else:
            td["action"].requires_grad = True
        with pytest.raises(
            RuntimeError,
            match="tensordict stored action requires grad.",
        ):
            _ = loss_fn._log_probs(td)
        if composite_action_dist:
            td["action"].apply_(lambda x: set_requires_grad(x, False))
        else:
            td["action"].requires_grad = False

        td = td.exclude(loss_fn.tensor_keys.value_target)
        loss = loss_fn(td)
        loss_critic = loss["loss_critic"]
        loss_objective = loss["loss_objective"] + loss.get("loss_entropy", 0.0)
        loss_critic.backward(retain_graph=True)
        # check that grads are independent and non null
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if separate_losses:
                if p.grad is not None and p.grad.norm() > 0.0:
                    assert "actor" not in name
                    assert "critic" in name
                if p.grad is None:
                    assert ("actor" in name) or ("target_" in name)
                    assert ("critic" not in name) or ("target_" in name)
            else:
                if p.grad is not None and p.grad.norm() > 0.0:
                    assert ("actor" in name) or ("critic" in name)
                if p.grad is None:
                    assert ("actor" in name) or ("critic" in name)

        critic.zero_grad()
        loss_objective.backward()
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" in name
                assert "critic" not in name
            if p.grad is None:
                assert ("actor" not in name) or ("target_" in name)
                assert ("critic" in name) or ("target_" in name)
        actor.zero_grad()

        # test reset
        loss_fn.reset()

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found, {FUNCTORCH_ERR}"
    )
    @pytest.mark.parametrize("gradient_mode", (True, False))
    @pytest.mark.parametrize("advantage", ("gae", "vtrace", "td", "td_lambda", None))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_a2c_diff(self, device, gradient_mode, advantage, composite_action_dist):
        if pack_version.parse(torch.__version__) > pack_version.parse("1.14"):
            raise pytest.skip("make_functional_with_buffers needs to be changed")
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_a2c(
            device=device, composite_action_dist=composite_action_dist
        )

        actor = self._create_mock_actor(
            device=device, composite_action_dist=composite_action_dist
        )
        value = self._create_mock_value(device=device)
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage == "td":
            advantage = TD1Estimator(
                gamma=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage == "vtrace":
            advantage = VTrace(
                gamma=0.9,
                value_network=value,
                actor_network=actor,
                differentiable=gradient_mode,
            )
        elif advantage == "td_lambda":
            advantage = TDLambdaEstimator(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage is None:
            pass
        else:
            raise NotImplementedError

        loss_fn = A2CLoss(actor, value, loss_critic_type="l2")

        floss_fn, params, buffers = make_functional_with_buffers(loss_fn)

        if advantage is not None:
            advantage(td)
        loss = floss_fn(params, buffers, td)
        loss_critic = loss["loss_critic"]
        loss_objective = loss["loss_objective"] + loss.get("loss_entropy", 0.0)
        loss_critic.backward(retain_graph=True)
        # check that grads are independent and non null
        named_parameters = loss_fn.named_parameters()
        for (name, _), p in zip(named_parameters, params):
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" not in name
                assert "critic" in name
            if p.grad is None:
                assert ("actor" in name) or ("target_" in name)
                assert ("critic" not in name) or ("target_" in name)

        for param in params:
            param.grad = None
        loss_objective.backward()
        named_parameters = loss_fn.named_parameters()
        for (name, _), p in zip(named_parameters, params):
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" in name
                assert "critic" not in name
            if p.grad is None:
                assert ("actor" not in name) or ("target_" in name)
                assert ("critic" in name) or ("target_" in name)
        for param in params:
            param.grad = None

    @pytest.mark.parametrize(
        "td_est",
        [
            ValueEstimators.TD1,
            ValueEstimators.TD0,
            ValueEstimators.GAE,
            ValueEstimators.VTrace,
            ValueEstimators.TDLambda,
        ],
    )
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_a2c_tensordict_keys(self, td_est, composite_action_dist):
        actor = self._create_mock_actor(composite_action_dist=composite_action_dist)
        value = self._create_mock_value()

        loss_fn = A2CLoss(actor, value, loss_critic_type="l2")

        default_keys = {
            "advantage": "advantage",
            "value_target": "value_target",
            "value": "state_value",
            "action": "action",
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
            "sample_log_prob": "action_log_prob",
        }

        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
            td_est=td_est,
        )

        value = self._create_mock_value(out_keys=["value_state_test"])

        loss_fn = A2CLoss(actor, value, loss_critic_type="l2")

        key_mapping = {
            "advantage": ("advantage", "advantage_test"),
            "value_target": ("value_target", "value_target_test"),
            "value": ("value", "value_state_test"),
            "reward": ("reward", "reward_test"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize(
        "td_est",
        [
            ValueEstimators.GAE,
            ValueEstimators.VTrace,
        ],
    )
    @pytest.mark.parametrize("advantage", ("gae", "vtrace", None))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_a2c_tensordict_keys_run(
        self, device, advantage, td_est, composite_action_dist
    ):
        """Test A2C loss module with non-default tensordict keys."""
        torch.manual_seed(self.seed)
        gradient_mode = True
        advantage_key = "advantage_test"
        value_target_key = "value_target_test"
        value_key = "state_value_test"
        action_key = "action_test"
        reward_key = "reward_test"
        sample_log_prob_key = "action_log_prob_test"
        done_key = ("done", "test")
        terminated_key = ("terminated", "test")

        td = self._create_seq_mock_data_a2c(
            device=device,
            action_key=action_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
            sample_log_prob_key=sample_log_prob_key,
            composite_action_dist=composite_action_dist,
        )

        actor = self._create_mock_actor(
            device=device,
            sample_log_prob_key=sample_log_prob_key,
            composite_action_dist=composite_action_dist,
            action_key=action_key,
        )
        value = self._create_mock_value(device=device, out_keys=[value_key])
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage == "vtrace":
            advantage = VTrace(
                gamma=0.9,
                value_network=value,
                actor_network=actor,
                differentiable=gradient_mode,
            )
        elif advantage is None:
            pass
        else:
            raise NotImplementedError

        loss_fn = A2CLoss(actor, value, loss_critic_type="l2")
        loss_fn.set_keys(
            advantage=advantage_key,
            value_target=value_target_key,
            value=value_key,
            action=action_key,
            reward=reward_key,
            done=done_key,
            terminated=done_key,
            sample_log_prob=sample_log_prob_key,
        )

        if advantage is not None:
            advantage.set_keys(
                advantage=advantage_key,
                value_target=value_target_key,
                value=value_key,
                reward=reward_key,
                done=done_key,
                terminated=terminated_key,
                sample_log_prob=sample_log_prob_key,
            )
            advantage(td)
        else:
            if td_est is not None:
                loss_fn.make_value_estimator(td_est)

        loss = loss_fn(td)
        loss_critic = loss["loss_critic"]
        loss_objective = loss["loss_objective"] + loss.get("loss_entropy", 0.0)
        loss_critic.backward(retain_graph=True)
        # check that grads are independent and non null
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" not in name
            if p.grad is None:
                assert ("actor" in name) or ("target_" in name)
                assert ("critic" not in name) or ("target_" in name)

        value.zero_grad()
        loss_objective.backward()
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" in name
                assert "critic" not in name
            if p.grad is None:
                assert ("actor" not in name) or ("target_" in name)
                assert ("critic" in name) or ("target_" in name)
        actor.zero_grad()

        # test reset
        loss_fn.reset()

    @pytest.mark.parametrize("action_key", ["action", "action2"])
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    @pytest.mark.parametrize(
        "composite_action_dist",
        [
            False,
        ],
    )
    def test_a2c_notensordict(
        self,
        action_key,
        observation_key,
        reward_key,
        done_key,
        terminated_key,
        composite_action_dist,
    ):
        torch.manual_seed(self.seed)

        actor = self._create_mock_actor(
            observation_key=observation_key, composite_action_dist=composite_action_dist
        )
        value = self._create_mock_value(observation_key=observation_key)
        td = self._create_seq_mock_data_a2c(
            action_key=action_key,
            observation_key=observation_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
            composite_action_dist=composite_action_dist,
        )

        loss = A2CLoss(actor, value)
        loss.set_keys(
            action=action_key,
            reward=reward_key,
            done=done_key,
            terminated=terminated_key,
        )

        kwargs = {
            observation_key: td.get(observation_key),
            f"next_{observation_key}": td.get(observation_key),
            f"next_{reward_key}": td.get(("next", reward_key)),
            f"next_{done_key}": td.get(("next", done_key)),
            f"next_{terminated_key}": td.get(("next", terminated_key)),
            action_key: td.get(action_key),
        }
        td = TensorDict(kwargs, td.batch_size).unflatten_keys("_")

        loss_val = loss(**kwargs)
        loss_val_td = loss(td)

        torch.testing.assert_close(loss_val_td.get("loss_objective"), loss_val[0])
        torch.testing.assert_close(loss_val_td.get("loss_critic"), loss_val[1])
        # don't test entropy and loss_entropy, since they depend on a random sample
        # from distribution
        assert len(loss_val) == 4
        # test select
        torch.manual_seed(self.seed)
        loss.select_out_keys("loss_objective", "loss_critic")
        if torch.__version__ >= "2.0.0":
            loss_objective, loss_critic = loss(**kwargs)
        else:
            with pytest.raises(
                RuntimeError,
                match="You are likely using tensordict.nn.dispatch with keyword arguments",
            ):
                loss_obj, loss_crit = loss(**kwargs)
            return
        assert loss_objective == loss_val_td["loss_objective"]
        assert loss_critic == loss_val_td["loss_critic"]

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_a2c_reduction(self, reduction, composite_action_dist):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        td = self._create_seq_mock_data_a2c(
            device=device, composite_action_dist=composite_action_dist
        )
        actor = self._create_mock_actor(
            device=device, composite_action_dist=composite_action_dist
        )
        value = self._create_mock_value(device=device)
        advantage = GAE(
            gamma=0.9,
            lmbda=0.9,
            value_network=value,
        )
        loss_fn = A2CLoss(
            actor,
            value,
            loss_critic_type="l2",
            reduction=reduction,
        )
        advantage(td)
        loss = loss_fn(td)
        if reduction == "none":
            for key in loss.keys():
                if key.startswith("loss_"):
                    assert loss[key].shape == td.shape
        else:
            for key in loss.keys():
                if not key.startswith("loss_"):
                    continue
                assert loss[key].shape == torch.Size([])

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("clip_value", [True, None, 0.5, torch.tensor(0.5)])
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_a2c_value_clipping(self, clip_value, device, composite_action_dist):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_a2c(
            device=device, composite_action_dist=composite_action_dist
        )
        actor = self._create_mock_actor(
            device=device, composite_action_dist=composite_action_dist
        )
        value = self._create_mock_value(device=device)
        advantage = GAE(
            gamma=0.9,
            lmbda=0.9,
            value_network=value,
        )

        if isinstance(clip_value, bool):
            with pytest.raises(
                ValueError,
                match=f"clip_value must be a float or a scalar tensor, got {clip_value}.",
            ):
                loss_fn = A2CLoss(
                    actor,
                    value,
                    loss_critic_type="l2",
                    clip_value=clip_value,
                )
        else:
            loss_fn = A2CLoss(
                actor,
                value,
                loss_critic_type="l2",
                clip_value=clip_value,
            )
            advantage(td)

            value = td.pop(loss_fn.tensor_keys.value)

            if clip_value:
                # Test it fails without value key
                with pytest.raises(
                    KeyError,
                    match=f"clip_value is set to {clip_value}, but the key "
                    "state_value was not found in the input tensordict. "
                    "Make sure that the value_key passed to A2C exists in "
                    "the input tensordict.",
                ):
                    loss = loss_fn(td)

            # Add value back to td
            td.set(loss_fn.tensor_keys.value, value)

            # Test it works with value key
            loss = loss_fn(td)
            assert "loss_critic" in loss.keys()


class TestReinforce(LossModuleTestBase):
    seed = 0

    def test_reset_parameters_recursive(self):
        n_obs = 3
        n_act = 5
        value_net = ValueOperator(nn.Linear(n_obs, 1), in_keys=["observation"])
        net = nn.Sequential(nn.Linear(n_obs, 2 * n_act), NormalParamExtractor())
        module = TensorDictModule(
            net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        actor_net = ProbabilisticActor(
            module,
            distribution_class=TanhNormal,
            return_log_prob=True,
            in_keys=["loc", "scale"],
            spec=Unbounded(n_act),
        )
        loss_fn = ReinforceLoss(
            actor_net,
            critic_network=value_net,
        )
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("gradient_mode", [True, False])
    @pytest.mark.parametrize("advantage", ["gae", "td", "td_lambda", None])
    @pytest.mark.parametrize(
        "td_est",
        [
            ValueEstimators.TD1,
            ValueEstimators.TD0,
            ValueEstimators.GAE,
            ValueEstimators.TDLambda,
            None,
        ],
    )
    @pytest.mark.parametrize(
        "delay_value,functional", [[False, True], [False, False], [True, True]]
    )
    def test_reinforce_value_net(
        self, advantage, gradient_mode, delay_value, td_est, functional
    ):
        n_obs = 3
        n_act = 5
        batch = 4
        gamma = 0.9
        value_net = ValueOperator(nn.Linear(n_obs, 1), in_keys=["observation"])
        net = nn.Sequential(nn.Linear(n_obs, 2 * n_act), NormalParamExtractor())
        module = TensorDictModule(
            net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        actor_net = ProbabilisticActor(
            module,
            distribution_class=TanhNormal,
            return_log_prob=True,
            in_keys=["loc", "scale"],
            spec=Unbounded(n_act),
        )
        if advantage == "gae":
            advantage = GAE(
                gamma=gamma,
                lmbda=0.9,
                value_network=value_net,
                differentiable=gradient_mode,
            )
        elif advantage == "td":
            advantage = TD1Estimator(
                gamma=gamma,
                value_network=value_net,
                differentiable=gradient_mode,
            )
        elif advantage == "td_lambda":
            advantage = TDLambdaEstimator(
                gamma=0.9,
                lmbda=0.9,
                value_network=value_net,
                differentiable=gradient_mode,
            )
        elif advantage is None:
            pass
        else:
            raise NotImplementedError

        loss_fn = ReinforceLoss(
            actor_net,
            critic_network=value_net,
            delay_value=delay_value,
            functional=functional,
        )

        td = TensorDict(
            {
                "observation": torch.randn(batch, n_obs),
                "next": {
                    "observation": torch.randn(batch, n_obs),
                    "reward": torch.randn(batch, 1),
                    "done": torch.zeros(batch, 1, dtype=torch.bool),
                    "terminated": torch.zeros(batch, 1, dtype=torch.bool),
                },
                "action": torch.randn(batch, n_act),
            },
            [batch],
            names=["time"],
        )
        with (
            pytest.warns(UserWarning, match="No target network updater has been")
            if delay_value and rl_warnings()
            else contextlib.nullcontext()
        ):
            if advantage is not None:
                params = TensorDict.from_module(value_net)
                if delay_value:
                    target_params = loss_fn.target_critic_network_params
                else:
                    target_params = None
                advantage(td, params=params, target_params=target_params)
            elif td_est is not None:
                loss_fn.make_value_estimator(td_est)
            loss_td = loss_fn(td)
            autograd.grad(
                loss_td.get("loss_actor"),
                actor_net.parameters(),
                retain_graph=True,
            )
            autograd.grad(
                loss_td.get("loss_value"),
                value_net.parameters(),
                retain_graph=True,
            )
            with pytest.raises(RuntimeError, match="One of the "):
                autograd.grad(
                    loss_td.get("loss_actor"),
                    value_net.parameters(),
                    retain_graph=True,
                    allow_unused=False,
                )
            with pytest.raises(RuntimeError, match="One of the "):
                autograd.grad(
                    loss_td.get("loss_value"),
                    actor_net.parameters(),
                    retain_graph=True,
                    allow_unused=False,
                )

    @pytest.mark.parametrize(
        "td_est",
        [
            ValueEstimators.TD1,
            ValueEstimators.TD0,
            ValueEstimators.GAE,
            ValueEstimators.TDLambda,
        ],
    )
    def test_reinforce_tensordict_keys(self, td_est):
        n_obs = 3
        n_act = 5
        value_net = ValueOperator(nn.Linear(n_obs, 1), in_keys=["observation"])
        net = nn.Sequential(nn.Linear(n_obs, 2 * n_act), NormalParamExtractor())
        module = TensorDictModule(
            net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        actor_net = ProbabilisticActor(
            module,
            distribution_class=TanhNormal,
            return_log_prob=True,
            in_keys=["loc", "scale"],
            spec=Unbounded(n_act),
        )

        loss_fn = ReinforceLoss(
            actor_net,
            critic_network=value_net,
        )

        default_keys = {
            "advantage": "advantage",
            "value_target": "value_target",
            "value": "state_value",
            "sample_log_prob": "action_log_prob",
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
        }

        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
            td_est=td_est,
        )

        value_net = ValueOperator(
            nn.Linear(n_obs, 1), in_keys=["observation"], out_keys=["state_value_test"]
        )

        loss_fn = ReinforceLoss(
            actor_net,
            critic_network=value_net,
        )

        key_mapping = {
            "advantage": ("advantage", "advantage_test"),
            "value_target": ("value_target", "value_target_test"),
            "value": ("value", "state_value_test"),
            "reward": ("reward", "reward_test"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize()
    def _create_mock_common_layer_setup(
        self, n_obs=3, n_act=4, ncells=4, batch=2, n_hidden=2, T=10
    ):
        common_net = MLP(
            num_cells=ncells,
            in_features=n_obs,
            depth=3,
            out_features=n_hidden,
        )
        actor_net = MLP(
            num_cells=ncells,
            in_features=n_hidden,
            depth=2,
            out_features=2 * n_act,
        )
        value_net = MLP(
            in_features=n_hidden,
            num_cells=ncells,
            depth=2,
            out_features=1,
        )
        batch = [batch, T]
        td = TensorDict(
            {
                "obs": torch.randn(*batch, n_obs),
                "action": torch.randn(*batch, n_act),
                "action_log_prob": torch.randn(*batch),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
                "terminated": torch.zeros(*batch, 1, dtype=torch.bool),
                "next": {
                    "obs": torch.randn(*batch, n_obs),
                    "reward": torch.randn(*batch, 1),
                    "done": torch.zeros(*batch, 1, dtype=torch.bool),
                    "terminated": torch.zeros(*batch, 1, dtype=torch.bool),
                },
            },
            batch,
            names=[None, "time"],
        )
        common = Mod(common_net, in_keys=["obs"], out_keys=["hidden"])
        actor = ProbSeq(
            common,
            Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
            Mod(NormalParamExtractor(), in_keys=["param"], out_keys=["loc", "scale"]),
            ProbMod(
                in_keys=["loc", "scale"],
                out_keys=["action"],
                distribution_class=TanhNormal,
            ),
        )
        critic = Seq(
            common, Mod(value_net, in_keys=["hidden"], out_keys=["state_value"])
        )
        return actor, critic, common, td

    @pytest.mark.parametrize("separate_losses", [False, True])
    def test_reinforce_tensordict_separate_losses(self, separate_losses):
        torch.manual_seed(self.seed)
        actor, critic, common, td = self._create_mock_common_layer_setup()
        loss_fn = ReinforceLoss(
            actor_network=actor,
            critic_network=critic,
            separate_losses=separate_losses,
        )

        loss = loss_fn(td)

        assert all(
            (p.grad is None) or (p.grad == 0).all()
            for p in loss_fn.critic_network_params.values(True, True)
        )
        assert all(
            (p.grad is None) or (p.grad == 0).all()
            for p in loss_fn.actor_network_params.values(True, True)
        )
        # check that losses are independent
        for k in loss.keys():
            # can't sum over loss_actor as it is a scalar ()
            if not k.startswith("loss") or k == "loss_actor":
                continue
            loss[k].sum().backward(retain_graph=True)
            if k == "loss_value":
                common_layers_no = len(list(common.parameters()))
                if separate_losses:
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(True, True)
                    )
                    common_layers = itertools.islice(
                        loss_fn.critic_network_params.values(True, True),
                        common_layers_no,
                    )
                    assert all(
                        (p.grad is None) or (p.grad == 0).all() for p in common_layers
                    )
                    critic_layers = itertools.islice(
                        loss_fn.critic_network_params.values(True, True),
                        common_layers_no,
                        None,
                    )
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all() for p in critic_layers
                    )
                else:
                    common_layers = itertools.islice(
                        loss_fn.critic_network_params.values(True, True),
                        common_layers_no,
                    )
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all() for p in common_layers
                    )
                    actor_layers = itertools.islice(
                        loss_fn.actor_network_params.values(True, True),
                        common_layers_no,
                        None,
                    )
                    assert all(
                        (p.grad is None) or (p.grad == 0).all() for p in actor_layers
                    )
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.critic_network_params.values(True, True)
                    )

            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()

    @pytest.mark.parametrize("action_key", ["action", "action2"])
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    def test_reinforce_notensordict(
        self, action_key, observation_key, reward_key, done_key, terminated_key
    ):
        torch.manual_seed(self.seed)
        n_obs = 3
        n_act = 5
        batch = 4
        value_net = ValueOperator(nn.Linear(n_obs, 1), in_keys=[observation_key])
        net = nn.Sequential(nn.Linear(n_obs, 2 * n_act), NormalParamExtractor())
        module = TensorDictModule(
            net, in_keys=[observation_key], out_keys=["loc", "scale"]
        )
        actor_net = ProbabilisticActor(
            module,
            distribution_class=TanhNormal,
            return_log_prob=True,
            in_keys=["loc", "scale"],
            spec=Unbounded(n_act),
        )
        loss = ReinforceLoss(actor_network=actor_net, critic_network=value_net)
        loss.set_keys(
            reward=reward_key,
            done=done_key,
            action=action_key,
            terminated=terminated_key,
        )

        observation = torch.randn(batch, n_obs)
        action = torch.randn(batch, n_act)
        next_reward = torch.randn(batch, 1)
        next_observation = torch.randn(batch, n_obs)
        next_done = torch.zeros(batch, 1, dtype=torch.bool)
        next_terminated = torch.zeros(batch, 1, dtype=torch.bool)

        kwargs = {
            action_key: action,
            observation_key: observation,
            f"next_{reward_key}": next_reward,
            f"next_{done_key}": next_done,
            f"next_{terminated_key}": next_terminated,
            f"next_{observation_key}": next_observation,
        }
        td = TensorDict(kwargs, [batch]).unflatten_keys("_")
        loss_val = loss(**kwargs)
        loss_val_td = loss(td)
        torch.testing.assert_close(loss_val_td.get("loss_actor"), loss_val[0])
        torch.testing.assert_close(loss_val_td.get("loss_value"), loss_val[1])
        # test select
        torch.manual_seed(self.seed)
        loss.select_out_keys("loss_actor")
        if torch.__version__ >= "2.0.0":
            loss_actor = loss(**kwargs)
        else:
            with pytest.raises(
                RuntimeError,
                match="You are likely using tensordict.nn.dispatch with keyword arguments",
            ):
                loss_actor = loss(**kwargs)
            return
        assert loss_actor == loss_val_td["loss_actor"]

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_reinforce_reduction(self, reduction):
        torch.manual_seed(self.seed)
        actor, critic, common, td = self._create_mock_common_layer_setup()
        loss_fn = ReinforceLoss(
            actor_network=actor,
            critic_network=critic,
            reduction=reduction,
        )
        loss = loss_fn(td)
        if reduction == "none":
            for key in loss.keys():
                if key.startswith("loss_"):
                    assert loss[key].shape == td.shape
        else:
            for key in loss.keys():
                if not key.startswith("loss_"):
                    continue
                assert loss[key].shape == torch.Size([])

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("clip_value", [True, None, 0.5, torch.tensor(0.5)])
    def test_reinforce_value_clipping(self, clip_value, device):
        torch.manual_seed(self.seed)
        actor, critic, common, td = self._create_mock_common_layer_setup()
        actor = actor.to(device)
        critic = critic.to(device)
        td = td.to(device)
        advantage = GAE(
            gamma=0.9,
            lmbda=0.9,
            value_network=critic,
        )
        if isinstance(clip_value, bool):
            with pytest.raises(
                ValueError,
                match=f"clip_value must be a float or a scalar tensor, got {clip_value}.",
            ):
                loss_fn = ReinforceLoss(
                    actor_network=actor,
                    critic_network=critic,
                    clip_value=clip_value,
                )
                return
        else:
            loss_fn = ReinforceLoss(
                actor_network=actor,
                critic_network=critic,
                clip_value=clip_value,
            )
            advantage(td)

            value = td.pop(loss_fn.tensor_keys.value)

            if clip_value:
                # Test it fails without value key
                with pytest.raises(
                    KeyError,
                    match=f"clip_value is set to {loss_fn.clip_value}, but the key "
                    "state_value was not found in the input tensordict. "
                    "Make sure that the value_key passed to Reinforce exists in "
                    "the input tensordict.",
                ):
                    loss = loss_fn(td)

            # Add value back to td
            td.set(loss_fn.tensor_keys.value, value)

            # Test it works with value key
            loss = loss_fn(td)
            assert "loss_value" in loss.keys()


@pytest.mark.parametrize("device", get_default_devices())
class TestDreamer(LossModuleTestBase):
    img_size = (64, 64)

    def _create_world_model_data(
        self, batch_size, temporal_length, rssm_hidden_dim, state_dim
    ):
        td = TensorDict(
            {
                "state": torch.zeros(batch_size, temporal_length, state_dim),
                "belief": torch.zeros(batch_size, temporal_length, rssm_hidden_dim),
                "pixels": torch.randn(batch_size, temporal_length, 3, *self.img_size),
                "next": {
                    "pixels": torch.randn(
                        batch_size, temporal_length, 3, *self.img_size
                    ),
                    "reward": torch.randn(batch_size, temporal_length, 1),
                    "done": torch.zeros(batch_size, temporal_length, dtype=torch.bool),
                    "terminated": torch.zeros(
                        batch_size, temporal_length, dtype=torch.bool
                    ),
                },
                "action": torch.randn(batch_size, temporal_length, 64),
            },
            [batch_size, temporal_length],
        )
        return td

    def _create_actor_data(
        self, batch_size, temporal_length, rssm_hidden_dim, state_dim
    ):
        td = TensorDict(
            {
                "state": torch.randn(batch_size, temporal_length, state_dim),
                "belief": torch.randn(batch_size, temporal_length, rssm_hidden_dim),
                "reward": torch.randn(batch_size, temporal_length, 1),
            },
            [batch_size, temporal_length],
        )
        return td

    def _create_value_data(
        self, batch_size, temporal_length, rssm_hidden_dim, state_dim
    ):
        td = TensorDict(
            {
                "state": torch.randn(batch_size * temporal_length, state_dim),
                "belief": torch.randn(batch_size * temporal_length, rssm_hidden_dim),
                "lambda_target": torch.randn(batch_size * temporal_length, 1),
            },
            [batch_size * temporal_length],
        )
        return td

    def _create_world_model_model(self, rssm_hidden_dim, state_dim, mlp_num_units=13):
        mock_env = TransformedEnv(
            ContinuousActionConvMockEnv(pixel_shape=[3, *self.img_size])
        )
        default_dict = {
            "state": Unbounded(state_dim),
            "belief": Unbounded(rssm_hidden_dim),
        }
        mock_env.append_transform(
            TensorDictPrimer(random=False, default_value=0, **default_dict)
        )

        obs_encoder = ObsEncoder(channels=3, num_layers=2)
        obs_decoder = ObsDecoder(channels=3, num_layers=4)

        rssm_prior = RSSMPrior(
            hidden_dim=rssm_hidden_dim,
            rnn_hidden_dim=rssm_hidden_dim,
            state_dim=state_dim,
            action_spec=mock_env.action_spec,
        )
        rssm_posterior = RSSMPosterior(hidden_dim=rssm_hidden_dim, state_dim=state_dim)

        # World Model and reward model
        rssm_rollout = RSSMRollout(
            TensorDictModule(
                rssm_prior,
                in_keys=["state", "belief", "action"],
                out_keys=[
                    ("next", "prior_mean"),
                    ("next", "prior_std"),
                    "_",
                    ("next", "belief"),
                ],
            ),
            TensorDictModule(
                rssm_posterior,
                in_keys=[("next", "belief"), ("next", "encoded_latents")],
                out_keys=[
                    ("next", "posterior_mean"),
                    ("next", "posterior_std"),
                    ("next", "state"),
                ],
            ),
        )
        reward_module = MLP(
            out_features=1, depth=2, num_cells=mlp_num_units, activation_class=nn.ELU
        )
        # World Model and reward model
        world_modeler = SafeSequential(
            TensorDictModule(
                obs_encoder,
                in_keys=[("next", "pixels")],
                out_keys=[("next", "encoded_latents")],
            ),
            rssm_rollout,
            TensorDictModule(
                obs_decoder,
                in_keys=[("next", "state"), ("next", "belief")],
                out_keys=[("next", "reco_pixels")],
            ),
        )
        reward_module = TensorDictModule(
            reward_module,
            in_keys=[("next", "state"), ("next", "belief")],
            out_keys=[("next", "reward")],
        )
        world_model = WorldModelWrapper(world_modeler, reward_module)

        with torch.no_grad():
            td = mock_env.rollout(10)
            td = td.unsqueeze(0).to_tensordict()
            td["state"] = torch.zeros((1, 10, state_dim))
            td["belief"] = torch.zeros((1, 10, rssm_hidden_dim))
            world_model(td)
        return world_model

    def _create_mb_env(self, rssm_hidden_dim, state_dim, mlp_num_units=13):
        mock_env = TransformedEnv(
            ContinuousActionConvMockEnv(pixel_shape=[3, *self.img_size])
        )
        default_dict = {
            "state": Unbounded(state_dim),
            "belief": Unbounded(rssm_hidden_dim),
        }
        mock_env.append_transform(
            TensorDictPrimer(random=False, default_value=0, **default_dict)
        )

        rssm_prior = RSSMPrior(
            hidden_dim=rssm_hidden_dim,
            rnn_hidden_dim=rssm_hidden_dim,
            state_dim=state_dim,
            action_spec=mock_env.action_spec,
        )
        reward_module = MLP(
            out_features=1, depth=2, num_cells=mlp_num_units, activation_class=nn.ELU
        )
        transition_model = SafeSequential(
            TensorDictModule(
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
        reward_model = TensorDictModule(
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
        )
        model_based_env.set_specs_from_env(mock_env)
        with torch.no_grad():
            model_based_env.rollout(3)
        return model_based_env

    def _create_actor_model(self, rssm_hidden_dim, state_dim, mlp_num_units=13):
        mock_env = TransformedEnv(
            ContinuousActionConvMockEnv(pixel_shape=[3, *self.img_size])
        )
        default_dict = {
            "state": Unbounded(state_dim),
            "belief": Unbounded(rssm_hidden_dim),
        }
        mock_env.append_transform(
            TensorDictPrimer(random=False, default_value=0, **default_dict)
        )

        actor_module = DreamerActor(
            out_features=mock_env.action_spec.shape[0],
            depth=1,
            num_cells=mlp_num_units,
            activation_class=nn.ELU,
        )
        actor_model = ProbabilisticTensorDictSequential(
            TensorDictModule(
                actor_module,
                in_keys=["state", "belief"],
                out_keys=["loc", "scale"],
            ),
            ProbabilisticTensorDictModule(
                in_keys=["loc", "scale"],
                out_keys="action",
                default_interaction_type=InteractionType.RANDOM,
                distribution_class=TanhNormal,
            ),
        )
        with torch.no_grad():
            td = TensorDict(
                {
                    "state": torch.randn(1, 2, state_dim),
                    "belief": torch.randn(1, 2, rssm_hidden_dim),
                },
                batch_size=[1],
            )
            actor_model(td)
        return actor_model

    def _create_value_model(self, rssm_hidden_dim, state_dim, mlp_num_units=13):
        value_model = TensorDictModule(
            MLP(
                out_features=1,
                depth=1,
                num_cells=mlp_num_units,
                activation_class=nn.ELU,
            ),
            in_keys=["state", "belief"],
            out_keys=["state_value"],
        )
        with torch.no_grad():
            td = TensorDict(
                {
                    "state": torch.randn(1, 2, state_dim),
                    "belief": torch.randn(1, 2, rssm_hidden_dim),
                },
                batch_size=[1],
            )
            value_model(td)
        return value_model

    def test_reset_parameters_recursive(self, device):
        world_model = self._create_world_model_model(10, 5).to(device)
        loss_fn = DreamerModelLoss(world_model)
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("lambda_kl", [0, 1.0])
    @pytest.mark.parametrize("lambda_reco", [0, 1.0])
    @pytest.mark.parametrize("lambda_reward", [0, 1.0])
    @pytest.mark.parametrize("reco_loss", ["l2", "smooth_l1"])
    @pytest.mark.parametrize("reward_loss", ["l2", "smooth_l1"])
    @pytest.mark.parametrize("free_nats", [-1000, 1000])
    @pytest.mark.parametrize("delayed_clamp", [False, True])
    def test_dreamer_world_model(
        self,
        device,
        lambda_reward,
        lambda_kl,
        lambda_reco,
        reward_loss,
        reco_loss,
        delayed_clamp,
        free_nats,
    ):
        tensordict = self._create_world_model_data(
            batch_size=2, temporal_length=3, rssm_hidden_dim=10, state_dim=5
        ).to(device)
        world_model = self._create_world_model_model(10, 5).to(device)
        loss_module = DreamerModelLoss(
            world_model,
            lambda_reco=lambda_reco,
            lambda_kl=lambda_kl,
            lambda_reward=lambda_reward,
            reward_loss=reward_loss,
            reco_loss=reco_loss,
            delayed_clamp=delayed_clamp,
            free_nats=free_nats,
        )
        loss_td, _ = loss_module(tensordict)
        for loss_str, lmbda in zip(
            ["loss_model_kl", "loss_model_reco", "loss_model_reward"],
            [lambda_kl, lambda_reco, lambda_reward],
        ):
            assert loss_td.get(loss_str) is not None
            assert loss_td.get(loss_str).shape == torch.Size([1])
            if lmbda == 0:
                assert loss_td.get(loss_str) == 0
            else:
                assert loss_td.get(loss_str) > 0

        loss = (
            loss_td.get("loss_model_kl")
            + loss_td.get("loss_model_reco")
            + loss_td.get("loss_model_reward")
        )
        loss.backward()
        grad_total = 0.0
        for name, param in loss_module.named_parameters():
            if param.grad is not None:
                valid_gradients = not (
                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                )
                if not valid_gradients:
                    raise ValueError(f"Invalid gradients for {name}")
                gsq = param.grad.pow(2).sum()
                grad_total += gsq.item()
        grad_is_zero = grad_total == 0
        if free_nats < 0:
            lambda_kl_corr = lambda_kl
        else:
            # we expect the kl loss to have 0 grad
            lambda_kl_corr = 0
        if grad_is_zero and (lambda_kl_corr or lambda_reward or lambda_reco):
            raise ValueError(
                f"Gradients are zero: lambdas={(lambda_kl_corr, lambda_reward, lambda_reco)}"
            )
        elif grad_is_zero:
            assert not (lambda_kl_corr or lambda_reward or lambda_reco)
        loss_module.zero_grad()

    @pytest.mark.parametrize("imagination_horizon", [3, 5])
    @pytest.mark.parametrize("discount_loss", [True, False])
    def test_dreamer_env(self, device, imagination_horizon, discount_loss):
        mb_env = self._create_mb_env(10, 5).to(device)
        rollout = mb_env.rollout(3)
        assert rollout.shape == torch.Size([3])
        # test reconstruction
        with pytest.raises(ValueError, match="No observation decoder provided"):
            mb_env.decode_obs(rollout)
        mb_env.obs_decoder = TensorDictModule(
            nn.LazyLinear(4, device=device),
            in_keys=["state"],
            out_keys=["reco_observation"],
        )
        # reconstruct
        mb_env.decode_obs(rollout)
        assert "reco_observation" in rollout.keys()
        # second pass
        tensordict = mb_env.decode_obs(mb_env.reset(), compute_latents=True)
        assert "reco_observation" in tensordict.keys()

    @pytest.mark.parametrize("imagination_horizon", [3, 5])
    @pytest.mark.parametrize("discount_loss", [True, False])
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_dreamer_actor(self, device, imagination_horizon, discount_loss, td_est):
        tensordict = self._create_actor_data(2, 3, 10, 5).to(device)
        mb_env = self._create_mb_env(10, 5).to(device)
        actor_model = self._create_actor_model(10, 5).to(device)
        value_model = self._create_value_model(10, 5).to(device)
        loss_module = DreamerActorLoss(
            actor_model,
            value_model,
            mb_env,
            imagination_horizon=imagination_horizon,
            discount_loss=discount_loss,
        )
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_module.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_module.make_value_estimator(td_est)
        loss_td, fake_data = loss_module(tensordict.reshape(-1))
        assert not fake_data.requires_grad
        assert fake_data.shape == torch.Size([tensordict.numel(), imagination_horizon])
        if discount_loss:
            assert loss_module.discount_loss

        assert loss_td.get("loss_actor") is not None
        loss = loss_td.get("loss_actor")
        loss.backward()
        grad_is_zero = True
        for name, param in loss_module.named_parameters():
            if param.grad is not None:
                valid_gradients = not (
                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                )
                grad_is_zero = (
                    grad_is_zero and torch.sum(torch.pow((param.grad), 2)) == 0
                )
                if not valid_gradients:
                    raise ValueError(f"Invalid gradients for {name}")
        if grad_is_zero:
            raise ValueError("Gradients are zero")
        loss_module.zero_grad()

    @pytest.mark.parametrize("discount_loss", [True, False])
    def test_dreamer_value(self, device, discount_loss):
        tensordict = self._create_value_data(2, 3, 10, 5).to(device)
        value_model = self._create_value_model(10, 5).to(device)
        loss_module = DreamerValueLoss(value_model, discount_loss=discount_loss)
        loss_td, fake_data = loss_module(tensordict)
        assert loss_td.get("loss_value") is not None
        assert not fake_data.requires_grad
        loss = loss_td.get("loss_value")
        loss.backward()
        grad_is_zero = True
        for name, param in loss_module.named_parameters():
            if param.grad is not None:
                valid_gradients = not (
                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                )
                grad_is_zero = (
                    grad_is_zero and torch.sum(torch.pow((param.grad), 2)) == 0
                )
                if not valid_gradients:
                    raise ValueError(f"Invalid gradients for {name}")
        if grad_is_zero:
            raise ValueError("Gradients are zero")
        loss_module.zero_grad()

    def test_dreamer_model_tensordict_keys(self, device):
        world_model = self._create_world_model_model(10, 5)
        loss_fn = DreamerModelLoss(world_model)

        default_keys = {
            "reward": "reward",
            "true_reward": "true_reward",
            "prior_mean": "prior_mean",
            "prior_std": "prior_std",
            "posterior_mean": "posterior_mean",
            "posterior_std": "posterior_std",
            "pixels": "pixels",
            "reco_pixels": "reco_pixels",
        }
        self.tensordict_keys_test(loss_fn, default_keys=default_keys)

    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_dreamer_actor_tensordict_keys(self, td_est, device):
        mb_env = self._create_mb_env(10, 5)
        actor_model = self._create_actor_model(10, 5)
        value_model = self._create_value_model(10, 5)
        loss_fn = DreamerActorLoss(
            actor_model,
            value_model,
            mb_env,
        )

        default_keys = {
            "belief": "belief",
            "reward": "reward",
            "value": "state_value",
            "done": "done",
            "terminated": "terminated",
        }
        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
            td_est=td_est,
        )

        loss_fn = DreamerActorLoss(
            actor_model,
            value_model,
            mb_env,
        )

        key_mapping = {"value": ("value", "value_test")}
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    def test_dreamer_value_tensordict_keys(self, device):
        value_model = self._create_value_model(10, 5)
        loss_fn = DreamerValueLoss(value_model)

        default_keys = {
            "value": "state_value",
        }
        self.tensordict_keys_test(loss_fn, default_keys=default_keys)


class TestOnlineDT(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = Bounded(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        net = nn.Sequential(nn.Linear(obs_dim, 2 * action_dim), NormalParamExtractor())
        module = TensorDictModule(
            net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        actor = ProbabilisticActor(
            module=module,
            distribution_class=TanhNormal,
            in_keys=["loc", "scale"],
            spec=action_spec,
        )
        return actor.to(device)

    def _create_mock_data_odt(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # create a tensordict
        obs = torch.randn(batch, obs_dim, device=device)
        action = torch.randn(batch, action_dim, device=device).clamp(-1, 1)
        reward2go = torch.randn(batch, 1, device=device)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "action": action,
                "reward2go": reward2go,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_odt(
        self, batch=2, T=4, obs_dim=3, action_dim=4, device="cpu"
    ):
        # create a tensordict
        obs = torch.randn(batch, T, obs_dim, device=device)
        action = torch.randn(batch, T, action_dim, device=device).clamp(-1, 1)
        reward2go = torch.randn(batch, T, 1, device=device)

        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs,
                "reward": reward2go,
                "action": action,
            },
            device=device,
        )
        return td

    def test_reset_parameters_recursive(self):
        actor = self._create_mock_actor()
        loss_fn = OnlineDTLoss(actor)
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("device", get_available_devices())
    def test_odt(self, device):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_odt(device=device)

        actor = self._create_mock_actor(device=device)

        loss_fn = OnlineDTLoss(actor)
        loss = loss_fn(td)
        loss_transformer = sum(
            loss[key]
            for key in loss.keys()
            if key.startswith("loss") and key != "loss_alpha"
        )
        loss_alpha = loss["loss_alpha"]
        loss_transformer.backward(retain_graph=True)
        named_parameters = loss_fn.named_parameters()

        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" in name
                assert "alpha" not in name
            if p.grad is None:
                assert "actor" not in name
                assert "alpha" in name
        loss_fn.zero_grad()
        loss_alpha.backward(retain_graph=True)
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" not in name
                assert "alpha" in name
            if p.grad is None:
                assert "actor" in name
                assert "alpha" not in name
        loss_fn.zero_grad()

        sum([loss_transformer, loss_alpha]).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has a null gradient"

    @pytest.mark.parametrize("device", get_available_devices())
    def test_odt_state_dict(self, device):
        torch.manual_seed(self.seed)

        actor = self._create_mock_actor(device=device)

        loss_fn = OnlineDTLoss(actor)
        sd = loss_fn.state_dict()
        loss_fn2 = OnlineDTLoss(actor)
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("action_dim", [1, 2, 4, 8])
    def test_odt_target_entropy_auto(self, action_dim):
        """Regression test for target_entropy='auto' should be -dim(A)."""
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(action_dim=action_dim)

        loss_fn = OnlineDTLoss(actor)
        # target_entropy="auto" should compute -action_dim
        assert (
            loss_fn.target_entropy.item() == -action_dim
        ), f"target_entropy should be -{action_dim}, got {loss_fn.target_entropy.item()}"

    @pytest.mark.parametrize("device", get_available_devices())
    def test_seq_odt(self, device):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_odt(device=device)

        actor = self._create_mock_actor(device=device)

        loss_fn = OnlineDTLoss(actor)
        loss = loss_fn(td)
        loss_transformer = sum(
            loss[key]
            for key in loss.keys()
            if key.startswith("loss") and key != "loss_alpha"
        )
        loss_alpha = loss["loss_alpha"]
        loss_transformer.backward(retain_graph=True)
        named_parameters = loss_fn.named_parameters()

        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" in name
                assert "alpha" not in name
            if p.grad is None:
                assert "actor" not in name
                assert "alpha" in name
        loss_fn.zero_grad()
        loss_alpha.backward(retain_graph=True)
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" not in name
                assert "alpha" in name
            if p.grad is None:
                assert "actor" in name
                assert "alpha" not in name
        loss_fn.zero_grad()

        sum([loss_transformer, loss_alpha]).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has a null gradient"

    def test_onlinedt_tensordict_keys(self):
        actor = self._create_mock_actor()
        loss_fn = OnlineDTLoss(actor)

        default_keys = {
            "action_pred": "action",
            "action_target": "action",
        }

        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
        )

    @pytest.mark.parametrize("device", get_default_devices())
    def test_onlinedt_notensordict(self, device):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)
        td = self._create_mock_data_odt(device=device)
        loss_fn = OnlineDTLoss(actor)

        in_keys = self._flatten_in_keys(loss_fn.in_keys)
        kwargs = dict(td.flatten_keys("_").select(*in_keys))

        torch.manual_seed(0)
        loss_val_td = loss_fn(td)
        torch.manual_seed(0)
        loss_log_likelihood, loss_entropy, loss_alpha, alpha, entropy = loss_fn(
            **kwargs
        )
        torch.testing.assert_close(
            loss_val_td.get("loss_log_likelihood"), loss_log_likelihood
        )
        torch.testing.assert_close(loss_val_td.get("loss_entropy"), loss_entropy)
        torch.testing.assert_close(loss_val_td.get("loss_alpha"), loss_alpha)
        # test select
        torch.manual_seed(0)
        loss_fn.select_out_keys("loss_entropy")
        if torch.__version__ >= "2.0.0":
            loss_entropy = loss_fn(**kwargs)
        else:
            with pytest.raises(
                RuntimeError,
                match="You are likely using tensordict.nn.dispatch with keyword arguments",
            ):
                loss_entropy = loss_fn(**kwargs)
            return
        assert loss_entropy == loss_val_td["loss_entropy"]

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_onlinedt_reduction(self, reduction):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        td = self._create_mock_data_odt(device=device)
        actor = self._create_mock_actor(device=device)
        loss_fn = OnlineDTLoss(
            actor,
            reduction=reduction,
            scalar_output_mode="exclude" if reduction == "none" else None,
        )
        loss = loss_fn(td)
        if reduction == "none":
            for key in loss.keys():
                if key.startswith("loss"):
                    assert loss[key].shape == td.shape
        else:
            for key in loss.keys():
                if not key.startswith("loss"):
                    continue
                assert loss[key].shape == torch.Size([])


class TestDT(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = Bounded(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        net = nn.Linear(obs_dim, action_dim)
        module = TensorDictModule(net, in_keys=["observation"], out_keys=["param"])
        actor = ProbabilisticActor(
            module=module,
            distribution_class=TanhDelta,
            in_keys=["param"],
            spec=action_spec,
        )
        return actor.to(device)

    def _create_mock_data_dt(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # create a tensordict
        obs = torch.randn(batch, obs_dim, device=device)
        action = torch.randn(batch, action_dim, device=device).clamp(-1, 1)
        reward2go = torch.randn(batch, 1, device=device)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "action": action,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_dt(
        self, batch=2, T=4, obs_dim=3, action_dim=4, device="cpu"
    ):
        # create a tensordict
        obs = torch.randn(batch, T, obs_dim, device=device)
        action = torch.randn(batch, T, action_dim, device=device).clamp(-1, 1)

        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs,
                "action": action,
            },
            device=device,
        )
        return td

    def test_reset_parameters_recursive(self):
        actor = self._create_mock_actor()
        loss_fn = DTLoss(actor)
        self.reset_parameters_recursive_test(loss_fn)

    def test_dt_tensordict_keys(self):
        actor = self._create_mock_actor()
        loss_fn = DTLoss(actor)

        default_keys = {
            "action_target": "action",
            "action_pred": "action",
        }

        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
        )

    @pytest.mark.parametrize("device", get_default_devices())
    def test_dt_notensordict(self, device):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)
        td = self._create_mock_data_dt(device=device)
        loss_fn = DTLoss(actor)

        in_keys = self._flatten_in_keys(loss_fn.in_keys)
        kwargs = dict(td.flatten_keys("_").select(*in_keys))

        loss_val_td = loss_fn(td)
        loss_val = loss_fn(**kwargs)
        torch.testing.assert_close(loss_val_td.get("loss"), loss_val)
        # test select
        loss_fn.select_out_keys("loss")
        if torch.__version__ >= "2.0.0":
            loss_actor = loss_fn(**kwargs)
        else:
            with pytest.raises(
                RuntimeError,
                match="You are likely using tensordict.nn.dispatch with keyword arguments",
            ):
                loss_actor = loss_fn(**kwargs)
            return
        assert loss_actor == loss_val_td["loss"]

    @pytest.mark.parametrize("device", get_available_devices())
    def test_dt(self, device):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_dt(device=device)

        actor = self._create_mock_actor(device=device)

        loss_fn = DTLoss(actor)
        loss = loss_fn(td)
        loss_transformer = loss["loss"]
        loss_transformer.backward(retain_graph=True)
        named_parameters = loss_fn.named_parameters()

        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" in name
                assert "alpha" not in name
            if p.grad is None:
                assert "actor" not in name
                assert "alpha" in name
        loss_fn.zero_grad()

        sum([loss_transformer]).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has a null gradient"

    @pytest.mark.parametrize("device", get_available_devices())
    def test_dt_state_dict(self, device):
        torch.manual_seed(self.seed)

        actor = self._create_mock_actor(device=device)

        loss_fn = DTLoss(actor)
        sd = loss_fn.state_dict()
        loss_fn2 = DTLoss(actor)
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("device", get_available_devices())
    def test_seq_dt(self, device):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_dt(device=device)

        actor = self._create_mock_actor(device=device)

        loss_fn = DTLoss(actor)
        loss = loss_fn(td)
        loss_transformer = loss["loss"]
        loss_transformer.backward(retain_graph=True)
        named_parameters = loss_fn.named_parameters()

        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" in name
                assert "alpha" not in name
            if p.grad is None:
                assert "actor" not in name
                assert "alpha" in name
        loss_fn.zero_grad()

        sum([loss_transformer]).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has a null gradient"

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_dt_reduction(self, reduction):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        td = self._create_mock_data_dt(device=device)
        actor = self._create_mock_actor(device=device)
        loss_fn = DTLoss(actor, reduction=reduction)
        loss = loss_fn(td)
        if reduction == "none":
            assert loss["loss"].shape == td["action"].shape
        else:
            assert loss["loss"].shape == torch.Size([])


class TestGAIL(LossModuleTestBase):
    seed = 0

    def _create_mock_discriminator(
        self, batch=2, obs_dim=3, action_dim=4, device="cpu"
    ):
        # Discriminator
        body = TensorDictModule(
            MLP(
                in_features=obs_dim + action_dim,
                out_features=32,
                depth=1,
                num_cells=32,
                activation_class=torch.nn.ReLU,
                activate_last_layer=True,
            ),
            in_keys=["observation", "action"],
            out_keys="hidden",
        )
        head = TensorDictModule(
            MLP(
                in_features=32,
                out_features=1,
                depth=0,
                num_cells=32,
                activation_class=torch.nn.Sigmoid,
                activate_last_layer=True,
            ),
            in_keys="hidden",
            out_keys="d_logits",
        )
        discriminator = TensorDictSequential(body, head)

        return discriminator.to(device)

    def _create_mock_data_gail(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # create a tensordict
        obs = torch.randn(batch, obs_dim, device=device)
        action = torch.randn(batch, action_dim, device=device).clamp(-1, 1)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "action": action,
                "collector_action": action,
                "collector_observation": obs,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_gail(
        self, batch=2, T=4, obs_dim=3, action_dim=4, device="cpu"
    ):
        # create a tensordict
        obs = torch.randn(batch, T, obs_dim, device=device)
        action = torch.randn(batch, T, action_dim, device=device).clamp(-1, 1)

        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs,
                "action": action,
                "collector_action": action,
                "collector_observation": obs,
            },
            device=device,
        )
        return td

    def test_reset_parameters_recursive(self):
        discriminator = self._create_mock_discriminator()
        loss_fn = GAILLoss(discriminator)
        self.reset_parameters_recursive_test(loss_fn)

    def test_gail_tensordict_keys(self):
        discriminator = self._create_mock_discriminator()
        loss_fn = GAILLoss(discriminator)

        default_keys = {
            "expert_action": "action",
            "expert_observation": "observation",
            "collector_action": "collector_action",
            "collector_observation": "collector_observation",
            "discriminator_pred": "d_logits",
        }

        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
        )

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("use_grad_penalty", [True, False])
    @pytest.mark.parametrize("gp_lambda", [0.1, 1.0])
    def test_gail_notensordict(self, device, use_grad_penalty, gp_lambda):
        torch.manual_seed(self.seed)
        discriminator = self._create_mock_discriminator(device=device)
        loss_fn = GAILLoss(
            discriminator, use_grad_penalty=use_grad_penalty, gp_lambda=gp_lambda
        )

        tensordict = self._create_mock_data_gail(device=device)

        in_keys = self._flatten_in_keys(loss_fn.in_keys)
        kwargs = dict(tensordict.flatten_keys("_").select(*in_keys))

        loss_val_td = loss_fn(tensordict)
        if use_grad_penalty:
            loss_val, _ = loss_fn(**kwargs)
        else:
            loss_val = loss_fn(**kwargs)

        torch.testing.assert_close(loss_val_td.get("loss"), loss_val)
        # test select
        loss_fn.select_out_keys("loss")
        if torch.__version__ >= "2.0.0":
            loss_discriminator = loss_fn(**kwargs)
        else:
            with pytest.raises(
                RuntimeError,
                match="You are likely using tensordict.nn.dispatch with keyword arguments",
            ):
                loss_discriminator = loss_fn(**kwargs)
            return
        assert loss_discriminator == loss_val_td["loss"]

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("use_grad_penalty", [True, False])
    @pytest.mark.parametrize("gp_lambda", [0.1, 1.0])
    def test_gail(self, device, use_grad_penalty, gp_lambda):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_gail(device=device)

        discriminator = self._create_mock_discriminator(device=device)

        loss_fn = GAILLoss(
            discriminator, use_grad_penalty=use_grad_penalty, gp_lambda=gp_lambda
        )
        loss = loss_fn(td)
        loss_transformer = loss["loss"]
        loss_transformer.backward(retain_graph=True)
        named_parameters = loss_fn.named_parameters()

        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "discriminator" in name
            if p.grad is None:
                assert "discriminator" not in name
        loss_fn.zero_grad()

        sum([loss_transformer]).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has a null gradient"

    @pytest.mark.parametrize("device", get_available_devices())
    def test_gail_state_dict(self, device):
        torch.manual_seed(self.seed)

        discriminator = self._create_mock_discriminator(device=device)

        loss_fn = GAILLoss(discriminator)
        sd = loss_fn.state_dict()
        loss_fn2 = GAILLoss(discriminator)
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("use_grad_penalty", [True, False])
    @pytest.mark.parametrize("gp_lambda", [0.1, 1.0])
    def test_seq_gail(self, device, use_grad_penalty, gp_lambda):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_gail(device=device)

        discriminator = self._create_mock_discriminator(device=device)

        loss_fn = GAILLoss(
            discriminator, use_grad_penalty=use_grad_penalty, gp_lambda=gp_lambda
        )
        loss = loss_fn(td)
        loss_transformer = loss["loss"]
        loss_transformer.backward(retain_graph=True)
        named_parameters = loss_fn.named_parameters()

        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "discriminator" in name
            if p.grad is None:
                assert "discriminator" not in name
        loss_fn.zero_grad()

        sum([loss_transformer]).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has a null gradient"

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    @pytest.mark.parametrize("use_grad_penalty", [True, False])
    @pytest.mark.parametrize("gp_lambda", [0.1, 1.0])
    def test_gail_reduction(self, reduction, use_grad_penalty, gp_lambda):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        td = self._create_mock_data_gail(device=device)
        discriminator = self._create_mock_discriminator(device=device)
        loss_fn = GAILLoss(discriminator, reduction=reduction)
        loss = loss_fn(td)
        if reduction == "none":
            assert loss["loss"].shape == (td["observation"].shape[0], 1)
        else:
            assert loss["loss"].shape == torch.Size([])


@pytest.mark.skipif(
    not _has_functorch, reason=f"functorch not installed: {FUNCTORCH_ERR}"
)
class TestIQL(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        observation_key="observation",
    ):
        # Actor
        action_spec = Bounded(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        net = nn.Sequential(nn.Linear(obs_dim, 2 * action_dim), NormalParamExtractor())
        module = TensorDictModule(
            net, in_keys=[observation_key], out_keys=["loc", "scale"]
        )
        actor = ProbabilisticActor(
            module=module,
            in_keys=["loc", "scale"],
            spec=action_spec,
            distribution_class=TanhNormal,
        )
        return actor.to(device)

    def _create_mock_qvalue(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        out_keys=None,
        observation_key="observation",
        action_key="action",
    ):
        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(obs_dim + action_dim, 1)

            def forward(self, obs, act):
                return self.linear(torch.cat([obs, act], -1))

        module = ValueClass()
        qvalue = ValueOperator(
            module=module,
            in_keys=[observation_key, action_key],
            out_keys=out_keys,
        )
        return qvalue.to(device)

    def _create_mock_value(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        out_keys=None,
        observation_key="observation",
    ):
        module = nn.Linear(obs_dim, 1)
        value = ValueOperator(
            module=module, in_keys=[observation_key], out_keys=out_keys
        )
        return value.to(device)

    def _create_mock_common_layer_setup(
        self, n_obs=3, n_act=4, ncells=4, batch=2, n_hidden=2, T=10
    ):
        common_net = MLP(
            num_cells=ncells,
            in_features=n_obs,
            depth=3,
            out_features=n_hidden,
        )
        actor_net = MLP(
            num_cells=ncells,
            in_features=n_hidden,
            depth=1,
            out_features=2 * n_act,
        )
        value_net = MLP(
            in_features=n_hidden,
            num_cells=ncells,
            depth=1,
            out_features=1,
        )
        qvalue_net = MLP(
            in_features=n_hidden + n_act,
            num_cells=ncells,
            depth=1,
            out_features=1,
        )
        batch = [batch, T]
        td = TensorDict(
            {
                "obs": torch.randn(*batch, n_obs),
                "action": torch.randn(*batch, n_act),
                "action_log_prob": torch.randn(*batch),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
                "terminated": torch.zeros(*batch, 1, dtype=torch.bool),
                "next": {
                    "obs": torch.randn(*batch, n_obs),
                    "reward": torch.randn(*batch, 1),
                    "done": torch.zeros(*batch, 1, dtype=torch.bool),
                    "terminated": torch.zeros(*batch, 1, dtype=torch.bool),
                },
            },
            batch,
            names=[None, "time"],
        )
        common = Mod(common_net, in_keys=["obs"], out_keys=["hidden"])
        actor = ProbSeq(
            common,
            Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
            Mod(NormalParamExtractor(), in_keys=["param"], out_keys=["loc", "scale"]),
            ProbMod(
                in_keys=["loc", "scale"],
                out_keys=["action"],
                distribution_class=TanhNormal,
            ),
        )
        value = Seq(
            common, Mod(value_net, in_keys=["hidden"], out_keys=["state_value"])
        )
        qvalue = Seq(
            common,
            Mod(
                qvalue_net,
                in_keys=["hidden", "action"],
                out_keys=["state_action_value"],
            ),
        )
        qvalue(actor(td.clone()))
        value(td.clone())
        return actor, value, qvalue, common, td

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_iql(
        self,
        batch=16,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
        observation_key="observation",
        action_key="action",
        done_key="done",
        terminated_key="terminated",
        reward_key="reward",
    ):
        # create a tensordict
        obs = torch.randn(batch, obs_dim, device=device)
        next_obs = torch.randn(batch, obs_dim, device=device)
        if atoms:
            raise NotImplementedError
        else:
            action = torch.randn(batch, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, 1, device=device)
        done = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch,),
            source={
                observation_key: obs,
                "next": {
                    observation_key: next_obs,
                    done_key: done,
                    terminated_key: terminated,
                    reward_key: reward,
                },
                action_key: action,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_iql(
        self, batch=8, T=4, obs_dim=3, action_dim=4, atoms=None, device="cpu"
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim, device=device)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action = torch.randn(batch, T, atoms, action_dim, device=device).clamp(
                -1, 1
            )
        else:
            action = torch.randn(batch, T, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = torch.ones(batch, T, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "next": {
                    "observation": next_obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                    "done": done,
                    "terminated": terminated,
                    "reward": reward.masked_fill_(~mask.unsqueeze(-1), 0.0),
                },
                "collector": {"mask": mask},
                "action": action.masked_fill_(~mask.unsqueeze(-1), 0.0),
            },
            names=[None, "time"],
            device=device,
        )
        return td

    def test_reset_parameters_recursive(self):
        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()
        value = self._create_mock_value()
        loss_fn = IQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
        )
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("temperature", [0.0, 0.1, 1.0, 10.0])
    @pytest.mark.parametrize("expectile", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_iql(
        self,
        num_qvalue,
        device,
        temperature,
        expectile,
        td_est,
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_iql(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        value = self._create_mock_value(device=device)

        loss_fn = IQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            temperature=temperature,
            expectile=expectile,
            loss_function="l2",
        )
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)

        with _check_td_steady(td), pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
            loss = loss_fn(td)
        assert loss_fn.tensor_keys.priority in td.keys()

        # check that losses are independent
        for k in loss.keys():
            if not k.startswith("loss"):
                continue
            loss[k].sum().backward(retain_graph=True)
            if k == "loss_actor":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            elif k == "loss_value":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            elif k == "loss_qvalue":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()

        sum(
            [item for name, item in loss.items() if name.startswith("loss_")]
        ).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            if not name.startswith("target_"):
                assert (
                    p.grad is not None and p.grad.norm() > 0.0
                ), f"parameter {name} (shape: {p.shape}) has a null gradient"
            else:
                assert (
                    p.grad is None or p.grad.norm() == 0.0
                ), f"target parameter {name} (shape: {p.shape}) has a non-null gradient"

    @pytest.mark.parametrize("num_qvalue", [2])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("temperature", [0.1])
    @pytest.mark.parametrize("expectile", [0.1])
    @pytest.mark.parametrize("td_est", [None])
    def test_iql_deactivate_vmap(
        self,
        num_qvalue,
        device,
        temperature,
        expectile,
        td_est,
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_iql(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        value = self._create_mock_value(device=device)

        torch.manual_seed(0)
        loss_fn_vmap = IQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            temperature=temperature,
            expectile=expectile,
            loss_function="l2",
            deactivate_vmap=False,
        )
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn_vmap.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn_vmap.make_value_estimator(td_est)

        with _check_td_steady(td), pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
            torch.manual_seed(1)
            loss_vmap = loss_fn_vmap(td)

        torch.manual_seed(0)
        loss_fn_no_vmap = IQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            temperature=temperature,
            expectile=expectile,
            loss_function="l2",
            deactivate_vmap=True,
        )
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn_no_vmap.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn_no_vmap.make_value_estimator(td_est)

        with pytest.raises(
            NotImplementedError,
            match="This implementation is not supported for torch<2.7",
        ) if torch.__version__ < "2.7" else contextlib.nullcontext():
            with _check_td_steady(td), pytest.warns(
                UserWarning, match="No target network updater"
            ) if rl_warnings() else contextlib.nullcontext():
                torch.manual_seed(1)
                loss_no_vmap = loss_fn_no_vmap(td)
            assert_allclose_td(loss_vmap, loss_no_vmap)

    @pytest.mark.parametrize("num_qvalue", [2])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("temperature", [0.0])
    @pytest.mark.parametrize("expectile", [0.1])
    def test_iql_state_dict(
        self,
        num_qvalue,
        device,
        temperature,
        expectile,
    ):
        torch.manual_seed(self.seed)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        value = self._create_mock_value(device=device)

        loss_fn = IQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            temperature=temperature,
            expectile=expectile,
            loss_function="l2",
        )
        sd = loss_fn.state_dict()
        loss_fn2 = IQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            temperature=temperature,
            expectile=expectile,
            loss_function="l2",
        )
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("separate_losses", [False, True])
    def test_iql_separate_losses(self, separate_losses):
        torch.manual_seed(self.seed)
        actor, value, qvalue, common, td = self._create_mock_common_layer_setup()
        loss_fn = IQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            loss_function="l2",
            separate_losses=separate_losses,
        )
        with pytest.warns(
            UserWarning, match="No target network updater has been"
        ) if rl_warnings() else contextlib.nullcontext():
            loss = loss_fn(td)

            assert loss_fn.tensor_keys.priority in td.keys()

            # check that losses are independent
            for k in loss.keys():
                if not k.startswith("loss"):
                    continue
                loss[k].sum().backward(retain_graph=True)
                common_layers_no = len(list(common.parameters()))
                if k == "loss_actor":
                    if separate_losses:
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in loss_fn.value_network_params.values(
                                include_nested=True, leaves_only=True
                            )
                        )
                    else:
                        common_layers = itertools.islice(
                            loss_fn.value_network_params.values(True, True),
                            common_layers_no,
                        )
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in common_layers
                        )
                        value_layers = itertools.islice(
                            loss_fn.value_network_params.values(True, True),
                            common_layers_no,
                            None,
                        )
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in value_layers
                        )
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.qvalue_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                elif k == "loss_value":
                    if separate_losses:
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in loss_fn.actor_network_params.values(
                                include_nested=True, leaves_only=True
                            )
                        )
                    else:
                        common_layers = itertools.islice(
                            loss_fn.actor_network_params.values(True, True),
                            common_layers_no,
                        )
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in common_layers
                        )
                        actor_layers = itertools.islice(
                            loss_fn.actor_network_params.values(True, True),
                            common_layers_no,
                            None,
                        )
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in actor_layers
                        )
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.qvalue_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                    if separate_losses:
                        common_layers = itertools.islice(
                            loss_fn.value_network_params.values(True, True),
                            common_layers_no,
                        )
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in common_layers
                        )
                        value_layers = itertools.islice(
                            loss_fn.value_network_params.values(True, True),
                            common_layers_no,
                            None,
                        )
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in value_layers
                        )
                    else:
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in loss_fn.value_network_params.values(
                                include_nested=True, leaves_only=True
                            )
                        )
                elif k == "loss_qvalue":
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.value_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                    if separate_losses:
                        common_layers = itertools.islice(
                            loss_fn.qvalue_network_params.values(True, True),
                            common_layers_no,
                        )
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in common_layers
                        )
                        qvalue_layers = itertools.islice(
                            loss_fn.qvalue_network_params.values(True, True),
                            common_layers_no,
                            None,
                        )
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in qvalue_layers
                        )
                    else:
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in loss_fn.qvalue_network_params.values(
                                include_nested=True, leaves_only=True
                            )
                        )
                else:
                    raise NotImplementedError(k)
                loss_fn.zero_grad()

    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("temperature", [0.0, 0.1, 1.0, 10.0])
    @pytest.mark.parametrize("expectile", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_iql_batcher(
        self,
        n,
        num_qvalue,
        temperature,
        expectile,
        device,
        gamma=0.9,
    ):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_iql(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        value = self._create_mock_value(device=device)

        loss_fn = IQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            temperature=temperature,
            expectile=expectile,
            loss_function="l2",
        )

        ms = MultiStep(gamma=gamma, n_steps=n).to(device)

        td_clone = td.clone()
        ms_td = ms(td_clone)

        torch.manual_seed(0)
        np.random.seed(0)
        with _check_td_steady(ms_td), pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
            loss_ms = loss_fn(ms_td)
        assert loss_fn.tensor_keys.priority in ms_td.keys()

        # Remove warnings
        SoftUpdate(loss_fn, eps=0.5)

        with torch.no_grad():
            torch.manual_seed(0)  # log-prob is computed with a random action
            np.random.seed(0)
            loss = loss_fn(td)
        if n == 1:
            assert_allclose_td(td, ms_td.select(*list(td.keys(True, True))))
            _loss = sum(
                [item for name, item in loss.items() if name.startswith("loss_")]
            )
            _loss_ms = sum(
                [item for name, item in loss_ms.items() if name.startswith("loss_")]
            )
            assert (
                abs(_loss - _loss_ms) < 1e-3
            ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                assert_allclose_td(loss, loss_ms)
        sum(
            [item for name, item in loss_ms.items() if name.startswith("loss_")]
        ).backward()
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if not name.startswith("target_"):
                assert (
                    p.grad is not None and p.grad.norm() > 0.0
                ), f"parameter {name} (shape: {p.shape}) has a null gradient"
            else:
                assert (
                    p.grad is None or p.grad.norm() == 0.0
                ), f"target parameter {name} (shape: {p.shape}) has a non-null gradient"

        # Check param update effect on targets
        target_qvalue = [
            p.clone()
            for p in loss_fn.target_qvalue_network_params.values(
                include_nested=True, leaves_only=True
            )
        ]
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        target_qvalue2 = [
            p.clone()
            for p in loss_fn.target_qvalue_network_params.values(
                include_nested=True, leaves_only=True
            )
        ]
        if loss_fn.delay_qvalue:
            assert all(
                (p1 == p2).all() for p1, p2 in zip(target_qvalue, target_qvalue2)
            )
        else:
            assert not any(
                (p1 == p2).any() for p1, p2 in zip(target_qvalue, target_qvalue2)
            )

        # check that policy is updated after parameter update
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_iql_tensordict_keys(self, td_est):
        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()
        value = self._create_mock_value()

        loss_fn = IQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            loss_function="l2",
        )

        default_keys = {
            "priority": "td_error",
            "log_prob": "_log_prob",
            "action": "action",
            "state_action_value": "state_action_value",
            "value": "state_value",
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
        }

        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
            td_est=td_est,
        )

        value = self._create_mock_value(out_keys=["value_test"])
        loss_fn = IQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            loss_function="l2",
        )

        key_mapping = {
            "value": ("value", "value_test"),
            "done": ("done", "done_test"),
            "terminated": ("terminated", "terminated_test"),
            "reward": ("reward", ("reward", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize("action_key", ["action", "action2"])
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    def test_iql_notensordict(
        self, action_key, observation_key, reward_key, done_key, terminated_key
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_iql(
            action_key=action_key,
            observation_key=observation_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
        )

        actor = self._create_mock_actor(observation_key=observation_key)
        qvalue = self._create_mock_qvalue(
            observation_key=observation_key,
            action_key=action_key,
            out_keys=["state_action_value"],
        )
        value = self._create_mock_value(observation_key=observation_key)

        loss = IQLLoss(actor_network=actor, qvalue_network=qvalue, value_network=value)
        loss.set_keys(
            action=action_key,
            reward=reward_key,
            done=done_key,
            terminated=terminated_key,
        )

        kwargs = {
            action_key: td.get(action_key),
            observation_key: td.get(observation_key),
            f"next_{reward_key}": td.get(("next", reward_key)),
            f"next_{done_key}": td.get(("next", done_key)),
            f"next_{terminated_key}": td.get(("next", terminated_key)),
            f"next_{observation_key}": td.get(("next", observation_key)),
        }
        td = TensorDict(kwargs, td.batch_size).unflatten_keys("_")

        with pytest.warns(
            UserWarning,
            match="No target network updater has been associated with this loss module",
        ) if rl_warnings() else contextlib.nullcontext():
            loss_val = loss(**kwargs)
            loss_val_td = loss(td)
            assert len(loss_val) == 4
            torch.testing.assert_close(loss_val_td.get("loss_actor"), loss_val[0])
            torch.testing.assert_close(loss_val_td.get("loss_qvalue"), loss_val[1])
            torch.testing.assert_close(loss_val_td.get("loss_value"), loss_val[2])
            torch.testing.assert_close(loss_val_td.get("entropy"), loss_val[3])
            # test select
            torch.manual_seed(self.seed)
            loss.select_out_keys("loss_actor", "loss_value")
            if torch.__version__ >= "2.0.0":
                loss_actor, loss_value = loss(**kwargs)
            else:
                with pytest.raises(
                    RuntimeError,
                    match="You are likely using tensordict.nn.dispatch with keyword arguments",
                ):
                    loss_actor, loss_value = loss(**kwargs)
                return
            assert loss_actor == loss_val_td["loss_actor"]
            assert loss_value == loss_val_td["loss_value"]

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_iql_reduction(self, reduction):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        td = self._create_mock_data_iql(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        value = self._create_mock_value(device=device)

        loss_fn = IQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            loss_function="l2",
            reduction=reduction,
            scalar_output_mode="exclude" if reduction == "none" else None,
        )
        loss_fn.make_value_estimator()
        with _check_td_steady(td), pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
            loss = loss_fn(td)
        if reduction == "none":
            for key in loss.keys():
                if key.startswith("loss"):
                    assert loss[key].shape == td.shape
        else:
            for key in loss.keys():
                if not key.startswith("loss"):
                    continue
                assert loss[key].shape == torch.Size([])


@pytest.mark.skipif(
    not _has_functorch, reason=f"functorch not installed: {FUNCTORCH_ERR}"
)
class TestDiscreteIQL(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        observation_key="observation",
        action_key="action",
    ):
        # Actor
        action_spec = OneHot(action_dim)
        net = nn.Linear(obs_dim, action_dim)
        module = TensorDictModule(net, in_keys=[observation_key], out_keys=["logits"])
        actor = ProbabilisticActor(
            spec=action_spec,
            module=module,
            in_keys=["logits"],
            out_keys=[action_key],
            distribution_class=OneHotCategorical,
            return_log_prob=False,
        )
        return actor.to(device)

    def _create_mock_qvalue(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        out_keys=None,
        observation_key="observation",
    ):
        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(obs_dim, action_dim)

            def forward(self, obs):
                return self.linear(obs)

        module = ValueClass()
        qvalue = ValueOperator(
            module=module, in_keys=[observation_key], out_keys=["state_action_value"]
        )
        return qvalue.to(device)

    def _create_mock_value(
        self,
        batch=2,
        obs_dim=3,
        device="cpu",
        out_keys=None,
        observation_key="observation",
    ):
        module = nn.Linear(obs_dim, 1)
        value = ValueOperator(
            module=module, in_keys=[observation_key], out_keys=out_keys
        )
        return value.to(device)

    def _create_mock_common_layer_setup(
        self, n_obs=3, n_act=4, ncells=4, batch=2, n_hidden=2, T=10
    ):
        common_net = MLP(
            num_cells=ncells,
            in_features=n_obs,
            depth=3,
            out_features=n_hidden,
        )
        actor_net = MLP(
            num_cells=ncells,
            in_features=n_hidden,
            depth=1,
            out_features=2 * n_act,
        )
        value_net = MLP(
            in_features=n_hidden,
            num_cells=ncells,
            depth=1,
            out_features=1,
        )
        qvalue_net = MLP(
            in_features=n_hidden,
            num_cells=ncells,
            depth=1,
            out_features=n_act,
        )
        batch = [batch, T]
        td = TensorDict(
            {
                "obs": torch.randn(*batch, n_obs),
                "action": torch.randn(*batch, n_act),
                "action_log_prob": torch.randn(*batch),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
                "terminated": torch.zeros(*batch, 1, dtype=torch.bool),
                "next": {
                    "obs": torch.randn(*batch, n_obs),
                    "reward": torch.randn(*batch, 1),
                    "done": torch.zeros(*batch, 1, dtype=torch.bool),
                    "terminated": torch.zeros(*batch, 1, dtype=torch.bool),
                },
            },
            batch,
            names=[None, "time"],
        )
        common = Mod(common_net, in_keys=["obs"], out_keys=["hidden"])
        actor = ProbSeq(
            common,
            Mod(actor_net, in_keys=["hidden"], out_keys=["logits"]),
            ProbMod(
                in_keys=["logits"],
                out_keys=["action"],
                distribution_class=OneHotCategorical,
            ),
        )
        value = Seq(
            common, Mod(value_net, in_keys=["hidden"], out_keys=["state_value"])
        )
        qvalue = Seq(
            common,
            Mod(
                qvalue_net,
                in_keys=["hidden"],
                out_keys=["state_action_value"],
            ),
        )
        qvalue(actor(td.clone()))
        value(td.clone())
        return actor, value, qvalue, common, td

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_discrete_iql(
        self,
        batch=16,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
        observation_key="observation",
        action_key="action",
        done_key="done",
        terminated_key="terminated",
        reward_key="reward",
    ):
        # create a tensordict
        obs = torch.randn(batch, obs_dim, device=device)
        next_obs = torch.randn(batch, obs_dim, device=device)
        if atoms:
            raise NotImplementedError
        else:
            action_value = torch.randn(batch, action_dim, device=device)
            action = (action_value == action_value.max(-1, True)[0]).to(torch.long)
        reward = torch.randn(batch, 1, device=device)
        done = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch,),
            source={
                observation_key: obs,
                "next": {
                    observation_key: next_obs,
                    done_key: done,
                    terminated_key: terminated,
                    reward_key: reward,
                },
                action_key: action,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_discrete_iql(
        self, batch=8, T=4, obs_dim=3, action_dim=4, atoms=None, device="cpu"
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim, device=device)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action = torch.randn(batch, T, atoms, action_dim, device=device).clamp(
                -1, 1
            )
        else:
            action_value = torch.randn(batch, T, action_dim, device=device)
            action = (action_value == action_value.max(-1, True)[0]).to(torch.long)

        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = torch.ones(batch, T, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "next": {
                    "observation": next_obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                    "done": done,
                    "terminated": terminated,
                    "reward": reward.masked_fill_(~mask.unsqueeze(-1), 0.0),
                },
                "collector": {"mask": mask},
                "action": action.masked_fill_(~mask.unsqueeze(-1), 0.0),
            },
            names=[None, "time"],
            device=device,
        )
        return td

    def test_reset_parameters_recursive(self):
        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()
        value = self._create_mock_value()
        loss_fn = DiscreteIQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            action_space="one-hot",
        )
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("temperature", [0.0, 0.1, 1.0, 10.0])
    @pytest.mark.parametrize("expectile", [0.1, 0.5])
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_discrete_iql(
        self,
        num_qvalue,
        device,
        temperature,
        expectile,
        td_est,
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_discrete_iql(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        value = self._create_mock_value(device=device)

        loss_fn = DiscreteIQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            temperature=temperature,
            expectile=expectile,
            loss_function="l2",
            action_space="one-hot",
        )
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)

        with _check_td_steady(td), pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
            loss = loss_fn(td)
        assert loss_fn.tensor_keys.priority in td.keys()

        # check that losses are independent
        for k in loss.keys():
            if not k.startswith("loss"):
                continue
            loss[k].sum().backward(retain_graph=True)
            if k == "loss_actor":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            elif k == "loss_value":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            elif k == "loss_qvalue":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()

        sum(
            [item for name, item in loss.items() if name.startswith("loss_")]
        ).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            if not name.startswith("target_"):
                assert (
                    p.grad is not None and p.grad.norm() > 0.0
                ), f"parameter {name} (shape: {p.shape}) has a null gradient"
            else:
                assert (
                    p.grad is None or p.grad.norm() == 0.0
                ), f"target parameter {name} (shape: {p.shape}) has a non-null gradient"

    @pytest.mark.parametrize("num_qvalue", [2])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("temperature", [0.0])
    @pytest.mark.parametrize("expectile", [0.1])
    def test_discrete_iql_state_dict(
        self,
        num_qvalue,
        device,
        temperature,
        expectile,
    ):
        torch.manual_seed(self.seed)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        value = self._create_mock_value(device=device)

        loss_fn = DiscreteIQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            temperature=temperature,
            expectile=expectile,
            loss_function="l2",
            action_space="one-hot",
        )
        sd = loss_fn.state_dict()
        loss_fn2 = DiscreteIQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            temperature=temperature,
            expectile=expectile,
            loss_function="l2",
            action_space="one-hot",
        )
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("separate_losses", [False, True])
    def test_discrete_iql_separate_losses(self, separate_losses):
        torch.manual_seed(self.seed)
        actor, value, qvalue, common, td = self._create_mock_common_layer_setup()
        loss_fn = DiscreteIQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            loss_function="l2",
            separate_losses=separate_losses,
            action_space="one-hot",
        )
        with pytest.warns(
            UserWarning, match="No target network updater has been"
        ) if rl_warnings() else contextlib.nullcontext():
            loss = loss_fn(td)

            assert loss_fn.tensor_keys.priority in td.keys()

            # check that losses are independent
            for k in loss.keys():
                if not k.startswith("loss"):
                    continue
                loss[k].sum().backward(retain_graph=True)
                common_layers_no = len(list(common.parameters()))
                if k == "loss_actor":
                    if separate_losses:
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in loss_fn.value_network_params.values(
                                include_nested=True, leaves_only=True
                            )
                        )
                    else:
                        common_layers = itertools.islice(
                            loss_fn.value_network_params.values(True, True),
                            common_layers_no,
                        )
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in common_layers
                        )
                        value_layers = itertools.islice(
                            loss_fn.value_network_params.values(True, True),
                            common_layers_no,
                            None,
                        )
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in value_layers
                        )
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.qvalue_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                elif k == "loss_value":
                    if separate_losses:
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in loss_fn.actor_network_params.values(
                                include_nested=True, leaves_only=True
                            )
                        )
                    else:
                        common_layers = itertools.islice(
                            loss_fn.actor_network_params.values(True, True),
                            common_layers_no,
                        )
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in common_layers
                        )
                        actor_layers = itertools.islice(
                            loss_fn.actor_network_params.values(True, True),
                            common_layers_no,
                            None,
                        )
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in actor_layers
                        )
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.qvalue_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                    if separate_losses:
                        common_layers = itertools.islice(
                            loss_fn.value_network_params.values(True, True),
                            common_layers_no,
                        )
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in common_layers
                        )
                        value_layers = itertools.islice(
                            loss_fn.value_network_params.values(True, True),
                            common_layers_no,
                            None,
                        )
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in value_layers
                        )
                    else:
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in loss_fn.value_network_params.values(
                                include_nested=True, leaves_only=True
                            )
                        )
                elif k == "loss_qvalue":
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.value_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                    if separate_losses:
                        common_layers = itertools.islice(
                            loss_fn.qvalue_network_params.values(True, True),
                            common_layers_no,
                        )
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in common_layers
                        )
                        qvalue_layers = itertools.islice(
                            loss_fn.qvalue_network_params.values(True, True),
                            common_layers_no,
                            None,
                        )
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in qvalue_layers
                        )
                    else:
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in loss_fn.qvalue_network_params.values(
                                include_nested=True, leaves_only=True
                            )
                        )
                else:
                    raise NotImplementedError(k)
                loss_fn.zero_grad()

    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("temperature", [0.0, 0.1, 1.0, 10.0])
    @pytest.mark.parametrize("expectile", [0.1, 0.5])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_discrete_iql_batcher(
        self,
        n,
        num_qvalue,
        temperature,
        expectile,
        device,
        gamma=0.9,
    ):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_discrete_iql(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        value = self._create_mock_value(device=device)

        loss_fn = DiscreteIQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            temperature=temperature,
            expectile=expectile,
            loss_function="l2",
            action_space="one-hot",
        )

        ms = MultiStep(gamma=gamma, n_steps=n).to(device)

        td_clone = td.clone()
        ms_td = ms(td_clone)

        torch.manual_seed(0)
        np.random.seed(0)
        with _check_td_steady(ms_td), pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
            loss_ms = loss_fn(ms_td)
        assert loss_fn.tensor_keys.priority in ms_td.keys()

        SoftUpdate(loss_fn, eps=0.5)

        with torch.no_grad():
            torch.manual_seed(0)  # log-prob is computed with a random action
            np.random.seed(0)
            loss = loss_fn(td)
        if n == 1:
            assert_allclose_td(td, ms_td.select(*list(td.keys(True, True))))
            _loss = sum(
                [item for name, item in loss.items() if name.startswith("loss_")]
            )
            _loss_ms = sum(
                [item for name, item in loss_ms.items() if name.startswith("loss_")]
            )
            assert (
                abs(_loss - _loss_ms) < 1e-3
            ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                assert_allclose_td(loss, loss_ms)
        sum(
            [item for name, item in loss_ms.items() if name.startswith("loss_")]
        ).backward()
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if not name.startswith("target_"):
                assert (
                    p.grad is not None and p.grad.norm() > 0.0
                ), f"parameter {name} (shape: {p.shape}) has a null gradient"
            else:
                assert (
                    p.grad is None or p.grad.norm() == 0.0
                ), f"target parameter {name} (shape: {p.shape}) has a non-null gradient"

        # Check param update effect on targets
        target_qvalue = [
            p.clone()
            for p in loss_fn.target_qvalue_network_params.values(
                include_nested=True, leaves_only=True
            )
        ]
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        target_qvalue2 = [
            p.clone()
            for p in loss_fn.target_qvalue_network_params.values(
                include_nested=True, leaves_only=True
            )
        ]
        if loss_fn.delay_qvalue:
            assert all(
                (p1 == p2).all() for p1, p2 in zip(target_qvalue, target_qvalue2)
            )
        else:
            assert not any(
                (p1 == p2).any() for p1, p2 in zip(target_qvalue, target_qvalue2)
            )

        # check that policy is updated after parameter update
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_discrete_iql_tensordict_keys(self, td_est):
        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()
        value = self._create_mock_value()

        loss_fn = DiscreteIQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            loss_function="l2",
            action_space="one-hot",
        )

        default_keys = {
            "priority": "td_error",
            "log_prob": "_log_prob",
            "action": "action",
            "state_action_value": "state_action_value",
            "value": "state_value",
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
        }

        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
            td_est=td_est,
        )

        value = self._create_mock_value(out_keys=["value_test"])
        loss_fn = DiscreteIQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            loss_function="l2",
            action_space="one-hot",
        )

        key_mapping = {
            "value": ("value", "value_test"),
            "done": ("done", "done_test"),
            "terminated": ("terminated", "terminated_test"),
            "reward": ("reward", ("reward", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize("action_key", ["action", "action2"])
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    def test_discrete_iql_notensordict(
        self, action_key, observation_key, reward_key, done_key, terminated_key
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_discrete_iql(
            action_key=action_key,
            observation_key=observation_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
        )

        actor = self._create_mock_actor(observation_key=observation_key)
        qvalue = self._create_mock_qvalue(
            observation_key=observation_key,
            out_keys=["state_action_value"],
        )
        value = self._create_mock_value(observation_key=observation_key)

        loss = DiscreteIQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            action_space="one-hot",
        )
        loss.set_keys(
            action=action_key,
            reward=reward_key,
            done=done_key,
            terminated=terminated_key,
        )

        kwargs = {
            action_key: td.get(action_key),
            observation_key: td.get(observation_key),
            f"next_{reward_key}": td.get(("next", reward_key)),
            f"next_{done_key}": td.get(("next", done_key)),
            f"next_{terminated_key}": td.get(("next", terminated_key)),
            f"next_{observation_key}": td.get(("next", observation_key)),
        }
        td = TensorDict(kwargs, td.batch_size).unflatten_keys("_")

        with pytest.warns(
            UserWarning,
            match="No target network updater has been associated with this loss module",
        ) if rl_warnings() else contextlib.nullcontext():
            loss_val = loss(**kwargs)
            loss_val_td = loss(td)
            assert len(loss_val) == 4
            torch.testing.assert_close(loss_val_td.get("loss_actor"), loss_val[0])
            torch.testing.assert_close(loss_val_td.get("loss_qvalue"), loss_val[1])
            torch.testing.assert_close(loss_val_td.get("loss_value"), loss_val[2])
            torch.testing.assert_close(loss_val_td.get("entropy"), loss_val[3])
            # test select
            torch.manual_seed(self.seed)
            loss.select_out_keys("loss_actor", "loss_value")
            if torch.__version__ >= "2.0.0":
                loss_actor, loss_value = loss(**kwargs)
            else:
                with pytest.raises(
                    RuntimeError,
                    match="You are likely using tensordict.nn.dispatch with keyword arguments",
                ):
                    loss_actor, loss_value = loss(**kwargs)
                return
            assert loss_actor == loss_val_td["loss_actor"]
            assert loss_value == loss_val_td["loss_value"]

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_discrete_iql_reduction(self, reduction):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        td = self._create_mock_data_discrete_iql(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        value = self._create_mock_value(device=device)

        loss_fn = DiscreteIQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            loss_function="l2",
            action_space="one-hot",
            reduction=reduction,
            scalar_output_mode="exclude" if reduction == "none" else None,
        )
        loss_fn.make_value_estimator()
        with _check_td_steady(td), pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
            loss = loss_fn(td)
        if reduction == "none":
            for key in loss.keys():
                if key.startswith("loss"):
                    assert loss[key].shape == td.shape
        else:
            for key in loss.keys():
                if not key.startswith("loss"):
                    continue
                assert loss[key].shape == torch.Size([])


@pytest.mark.parametrize("create_target_params", [True, False])
@pytest.mark.parametrize(
    "cast", [None, torch.float, torch.double, *get_default_devices()]
)
def test_param_buffer_types(create_target_params, cast):
    class MyLoss(LossModule):
        actor_network: TensorDictModule
        actor_network_params: TensorDict
        target_actor_network_params: TensorDict

        def __init__(self, actor_network):
            super().__init__()
            self.convert_to_functional(
                actor_network,
                "actor_network",
                create_target_params=create_target_params,
            )

        def _forward_value_estimator_keys(self, **kwargs) -> None:
            pass

    actor_module = TensorDictModule(
        nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4)),
        in_keys=["obs"],
        out_keys=["action"],
    )
    loss = MyLoss(actor_module)

    LossModuleTestBase.reset_parameters_recursive_test(loss)

    if create_target_params:
        SoftUpdate(loss, eps=0.5)

    if cast is not None:
        loss.to(cast)
    for name in ("weight", "bias"):
        param = loss.actor_network_params["module", "0", name]
        assert isinstance(param, nn.Parameter)
        target = loss.target_actor_network_params["module", "0", name]
        if create_target_params:
            assert target.data_ptr() != param.data_ptr()
        else:
            assert target.data_ptr() == param.data_ptr()
        assert param.requires_grad
        assert not target.requires_grad
        if cast is not None:
            if isinstance(cast, torch.dtype):
                assert param.dtype == cast
                assert target.dtype == cast
            else:
                assert param.device == cast
                assert target.device == cast

    if create_target_params:
        assert (
            loss.actor_network_params["module", "0", "weight"].data.data_ptr()
            != loss.target_actor_network_params["module", "0", "weight"].data.data_ptr()
        )
        assert (
            loss.actor_network_params["module", "0", "bias"].data.data_ptr()
            != loss.target_actor_network_params["module", "0", "bias"].data.data_ptr()
        )
    else:
        assert (
            loss.actor_network_params["module", "0", "weight"].data.data_ptr()
            == loss.target_actor_network_params["module", "0", "weight"].data.data_ptr()
        )
        assert (
            loss.actor_network_params["module", "0", "bias"].data.data_ptr()
            == loss.target_actor_network_params["module", "0", "bias"].data.data_ptr()
        )

    assert loss.actor_network_params["module", "0", "bias"].requires_grad
    assert not loss.target_actor_network_params["module", "0", "bias"].requires_grad
    assert not isinstance(
        loss.actor_network_params["module", "1", "running_mean"], nn.Parameter
    )
    assert not isinstance(
        loss.target_actor_network_params["module", "1", "running_mean"], nn.Parameter
    )
    assert not isinstance(
        loss.actor_network_params["module", "1", "running_var"], nn.Parameter
    )
    assert not isinstance(
        loss.target_actor_network_params["module", "1", "running_var"], nn.Parameter
    )


def test_hold_out():
    net = torch.nn.Linear(3, 4)
    x = torch.randn(1, 3)
    x_rg = torch.randn(1, 3, requires_grad=True)
    y = net(x)
    assert y.requires_grad
    with hold_out_net(net):
        y = net(x)
        assert not y.requires_grad
        y = net(x_rg)
        assert y.requires_grad

    y = net(x)
    assert y.requires_grad

    # nested case
    with hold_out_net(net):
        y = net(x)
        assert not y.requires_grad
        with hold_out_net(net):
            y = net(x)
            assert not y.requires_grad
            y = net(x_rg)
            assert y.requires_grad

    y = net(x)
    assert y.requires_grad

    # exception
    net = torch.nn.Sequential()
    with hold_out_net(net):
        pass


@pytest.mark.parametrize("mode", ["hard", "soft"])
@pytest.mark.parametrize("value_network_update_interval", [100, 1000])
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float64,
        torch.float32,
    ],
)
def test_updater(mode, value_network_update_interval, device, dtype):
    torch.manual_seed(100)

    class custom_module_error(nn.Module):
        def __init__(self):
            super().__init__()
            self.target_params = [torch.randn(3, 4)]
            self.target_error_params = [torch.randn(3, 4)]
            self.params = nn.ParameterList(
                [nn.Parameter(torch.randn(3, 4, requires_grad=True))]
            )

    module = custom_module_error().to(device)
    with pytest.raises(
        ValueError, match="The loss_module must be a LossModule instance"
    ):
        if mode == "hard":
            upd = HardUpdate(
                module, value_network_update_interval=value_network_update_interval
            )
        elif mode == "soft":
            upd = SoftUpdate(module, eps=1 - 1 / value_network_update_interval)

    class custom_module(LossModule):
        module1: TensorDictModule
        module1_params: TensorDict
        target_module1_params: TensorDict

        def __init__(self, delay_module=True):
            super().__init__()
            module1 = torch.nn.BatchNorm2d(10).eval()
            self.convert_to_functional(
                module1, "module1", create_target_params=delay_module
            )

            module2 = torch.nn.BatchNorm2d(10).eval()
            self.module2 = module2
            tparam = self._modules.get("target_module1_params", None)
            if tparam is None:
                tparam = self._modules.get("module1_params").data
            iterator_params = tparam.values(include_nested=True, leaves_only=True)
            for target in iterator_params:
                if target.dtype is not torch.int64:
                    target.data.normal_()
                else:
                    target.data += 10

        def _forward_value_estimator_keys(self, **kwargs) -> None:
            pass

    module = custom_module(delay_module=False)
    with pytest.raises(
        RuntimeError,
        match="Did not find any target parameters or buffers in the loss module",
    ):
        if mode == "hard":
            upd = HardUpdate(
                module, value_network_update_interval=value_network_update_interval
            )
        elif mode == "soft":
            upd = SoftUpdate(
                module,
                eps=1 - 1 / value_network_update_interval,
            )
        else:
            raise NotImplementedError

    # this is now allowed
    # with pytest.warns(UserWarning, match="No target network updater has been"):
    #     module = custom_module().to(device).to(dtype)

    if mode == "soft":
        with pytest.raises(ValueError, match="One and only one argument"):
            upd = SoftUpdate(
                module,
                eps=1 - 1 / value_network_update_interval,
                tau=0.1,
            )

    module = custom_module(delay_module=True)
    _ = module.module1_params
    with pytest.warns(
        UserWarning, match="No target network updater has been"
    ) if rl_warnings() else contextlib.nullcontext():
        _ = module.target_module1_params
    if mode == "hard":
        upd = HardUpdate(
            module, value_network_update_interval=value_network_update_interval
        )
    elif mode == "soft":
        upd = SoftUpdate(module, eps=1 - 1 / value_network_update_interval)
    for _, _v in upd._targets.items(True, True):
        if _v.dtype is not torch.int64:
            _v.copy_(torch.randn_like(_v))
        else:
            _v += 10

    # total dist
    d0 = 0.0
    for key, source_val in upd._sources.items(True, True):
        if not isinstance(key, tuple):
            key = (key,)
        key = ("target_" + key[0], *key[1:])
        target_val = upd._targets[key]
        assert target_val.dtype is source_val.dtype, key
        assert target_val.device == source_val.device, key
        if target_val.dtype == torch.long:
            continue
        with torch.no_grad():
            d0 += (target_val - source_val).norm().item()

    assert d0 > 0
    if mode == "hard":
        for i in range(value_network_update_interval + 1):
            # test that no update is occuring until value_network_update_interval
            d1 = 0.0
            for key, source_val in upd._sources.items(True, True):
                if not isinstance(key, tuple):
                    key = (key,)
                key = ("target_" + key[0], *key[1:])
                target_val = upd._targets[key]
                if target_val.dtype == torch.long:
                    continue
                with torch.no_grad():
                    d1 += (target_val - source_val).norm().item()

            assert d1 == d0, i
            assert upd.counter == i
            upd.step()
        assert upd.counter == 0
        # test that a new update has occured
        d1 = 0.0
        for key, source_val in upd._sources.items(True, True):
            if not isinstance(key, tuple):
                key = (key,)
            key = ("target_" + key[0], *key[1:])
            target_val = upd._targets[key]
            if target_val.dtype == torch.long:
                continue
            with torch.no_grad():
                d1 += (target_val - source_val).norm().item()
        assert d1 < d0

    elif mode == "soft":
        upd.step()
        d1 = 0.0
        for key, source_val in upd._sources.items(True, True):
            if not isinstance(key, tuple):
                key = (key,)
            key = ("target_" + key[0], *key[1:])
            target_val = upd._targets[key]
            if target_val.dtype == torch.long:
                continue
            with torch.no_grad():
                d1 += (target_val - source_val).norm().item()
        assert d1 < d0
    with pytest.warns(UserWarning, match="already"):
        upd.init_()
    upd.step()
    d2 = 0.0
    for key, source_val in upd._sources.items(True, True):
        if not isinstance(key, tuple):
            key = (key,)
        key = ("target_" + key[0], *key[1:])
        target_val = upd._targets[key]
        if target_val.dtype == torch.long:
            continue
        with torch.no_grad():
            d2 += (target_val - source_val).norm().item()
    assert d2 < 1e-6


class TestValues:
    @pytest.mark.skipif(not _has_gym, reason="requires gym")
    def test_gae_multi_done(self):

        # constants
        batch_size = 10
        seq_size = 5
        n_dims = batch_size
        gamma = 0.99
        lmbda = 0.98

        env = SerialEnv(
            batch_size, [functools.partial(GymEnv, "CartPole-v1")] * batch_size
        )
        obs_size = env.full_observation_spec[env.observation_keys[0]].shape[-1]

        td = env.rollout(seq_size, break_when_any_done=False)
        # make the magic happen: swap dims and create an artificial ndim done state
        done = td["next", "done"].transpose(0, -1)
        terminated = td["next", "terminated"].transpose(0, -1)
        reward = td["next", "reward"].transpose(0, -1)
        td = td[:1]
        td["next", "done"] = done
        td["next", "terminated"] = terminated
        td["next", "reward"] = reward

        critic = TensorDictModule(
            nn.Linear(obs_size, n_dims),
            in_keys=[("observation",)],
            out_keys=[("state_value",)],
        )

        gae_shifted = GAE(gamma=gamma, lmbda=lmbda, value_network=critic, shifted=True)
        gae_no_shifted = GAE(
            gamma=gamma, lmbda=lmbda, value_network=critic, shifted=False
        )

        torch.testing.assert_close(
            gae_shifted(td.clone())["advantage"],
            gae_no_shifted(td.clone())["advantage"],
        )

    @pytest.mark.skipif(not _has_gym, reason="requires gym")
    @pytest.mark.parametrize("module", ["lstm", "gru"])
    def test_gae_recurrent(self, module):
        # Checks that shifted=True and False provide the same result in GAE when an LSTM is used
        env = SerialEnv(
            2,
            [
                functools.partial(
                    TransformedEnv, GymEnv(PENDULUM_VERSIONED()), InitTracker()
                )
                for _ in range(2)
            ],
        )
        env.set_seed(0)
        torch.manual_seed(0)
        if module == "lstm":
            recurrent_module = LSTMModule(
                input_size=env.observation_spec["observation"].shape[-1],
                hidden_size=64,
                in_keys=["observation", "rs_h", "rs_c"],
                out_keys=["intermediate", ("next", "rs_h"), ("next", "rs_c")],
                python_based=True,
                dropout=0,
            )
        elif module == "gru":
            recurrent_module = GRUModule(
                input_size=env.observation_spec["observation"].shape[-1],
                hidden_size=64,
                in_keys=["observation", "rs_h"],
                out_keys=["intermediate", ("next", "rs_h")],
                python_based=True,
                dropout=0,
            )
        else:
            raise NotImplementedError
        recurrent_module.eval()
        mlp_value = MLP(num_cells=[64], out_features=1)
        value_net = Seq(
            recurrent_module,
            Mod(mlp_value, in_keys=["intermediate"], out_keys=["state_value"]),
        )
        mlp_policy = MLP(num_cells=[64], out_features=1)
        policy_net = Seq(
            recurrent_module,
            Mod(mlp_policy, in_keys=["intermediate"], out_keys=["action"]),
        )
        env = env.append_transform(recurrent_module.make_tensordict_primer())
        vals = env.rollout(1000, policy_net, break_when_any_done=False)
        value_net(vals.copy())

        # Shifted
        gae_shifted = GAE(
            gamma=0.9,
            lmbda=0.99,
            value_network=value_net,
            shifted=True,
        )
        with set_recurrent_mode(True):
            r0 = gae_shifted(vals.copy())
        a0 = r0["advantage"]

        gae = GAE(
            gamma=0.9,
            lmbda=0.99,
            value_network=value_net,
            shifted=False,
            deactivate_vmap=True,
        )
        with pytest.raises(
            NotImplementedError,
            match="This implementation is not supported for torch<2.7",
        ) if torch.__version__ < "2.7" else contextlib.nullcontext():
            with set_recurrent_mode(True):
                r1 = gae(vals.copy())
            a1 = r1["advantage"]
            torch.testing.assert_close(a0, a1)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.1, 0.5, 0.99])
    @pytest.mark.parametrize("lmbda", [0.1, 0.5, 0.99])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [200, 5, 3])
    # @pytest.mark.parametrize("random_gamma,rolling_gamma", [[True, False], [True, True], [False, None]])
    @pytest.mark.parametrize("random_gamma,rolling_gamma", [[False, None]])
    def test_tdlambda(self, device, gamma, lmbda, N, T, random_gamma, rolling_gamma):
        torch.manual_seed(0)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated = done.clone().bernoulli_(0.1)
        done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        if random_gamma:
            gamma = torch.rand_like(reward) * gamma

        next_state_value = torch.cat(
            [state_value[..., 1:, :], torch.randn_like(state_value[..., -1:, :])], -2
        )
        r1 = vec_td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
        )
        r2 = td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
        )
        r3, *_ = vec_generalized_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        torch.testing.assert_close(r3, r2, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(r1, r3, rtol=1e-4, atol=1e-4)

        # test when v' is not v from next step (not working with gae)
        next_state_value = torch.randn_like(next_state_value)
        r1 = vec_td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
        )
        r2 = td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
        )
        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.1, 0.99])
    @pytest.mark.parametrize("lmbda", [0.1, 0.99])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [3, 100])
    @pytest.mark.parametrize("feature_dim", [[5], [2, 5]])
    @pytest.mark.parametrize("random_gamma,rolling_gamma", [[False, None]])
    def test_tdlambda_multi(
        self, device, gamma, lmbda, N, T, random_gamma, rolling_gamma, feature_dim
    ):
        torch.manual_seed(0)
        D = feature_dim
        time_dim = -1 - len(D)
        done = torch.zeros(*N, T, *D, device=device, dtype=torch.bool)
        terminated = done.clone().bernoulli_(0.1)
        done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, *D, device=device)
        state_value = torch.randn(*N, T, *D, device=device)
        next_state_value = torch.randn(*N, T, *D, device=device)
        if random_gamma:
            gamma = torch.rand_like(reward) * gamma

        r1 = vec_td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
            time_dim=time_dim,
        )
        r2 = td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
            time_dim=time_dim,
        )
        if len(D) == 2:
            r3 = torch.cat(
                [
                    vec_td_lambda_advantage_estimate(
                        gamma,
                        lmbda,
                        state_value[..., i : i + 1, j],
                        next_state_value[..., i : i + 1, j],
                        reward[..., i : i + 1, j],
                        done=done[..., i : i + 1, j],
                        terminated=terminated[..., i : i + 1, j],
                        rolling_gamma=rolling_gamma,
                        time_dim=-2,
                    )
                    for i in range(D[0])
                    for j in range(D[1])
                ],
                -1,
            ).unflatten(-1, D)
            r4 = torch.cat(
                [
                    td_lambda_advantage_estimate(
                        gamma,
                        lmbda,
                        state_value[..., i : i + 1, j],
                        next_state_value[..., i : i + 1, j],
                        reward[..., i : i + 1, j],
                        done=done[..., i : i + 1, j],
                        terminated=terminated[..., i : i + 1, j],
                        rolling_gamma=rolling_gamma,
                        time_dim=-2,
                    )
                    for i in range(D[0])
                    for j in range(D[1])
                ],
                -1,
            ).unflatten(-1, D)
        else:
            r3 = torch.cat(
                [
                    vec_td_lambda_advantage_estimate(
                        gamma,
                        lmbda,
                        state_value[..., i : i + 1],
                        next_state_value[..., i : i + 1],
                        reward[..., i : i + 1],
                        done=done[..., i : i + 1],
                        terminated=terminated[..., i : i + 1],
                        rolling_gamma=rolling_gamma,
                        time_dim=-2,
                    )
                    for i in range(D[0])
                ],
                -1,
            )
            r4 = torch.cat(
                [
                    td_lambda_advantage_estimate(
                        gamma,
                        lmbda,
                        state_value[..., i : i + 1],
                        next_state_value[..., i : i + 1],
                        reward[..., i : i + 1],
                        done=done[..., i : i + 1],
                        terminated=terminated[..., i : i + 1],
                        rolling_gamma=rolling_gamma,
                        time_dim=-2,
                    )
                    for i in range(D[0])
                ],
                -1,
            )

        torch.testing.assert_close(r4, r2, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(r3, r1, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.1, 0.5, 0.99])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [3, 100])
    @pytest.mark.parametrize("random_gamma,rolling_gamma", [[False, None]])
    def test_td1(self, device, gamma, N, T, random_gamma, rolling_gamma):
        torch.manual_seed(0)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated = done.clone().bernoulli_(0.1)
        done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)
        if random_gamma:
            gamma = torch.rand_like(reward) * gamma

        r1 = vec_td1_advantage_estimate(
            gamma,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
        )
        r2 = td1_advantage_estimate(
            gamma,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
        )
        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.1, 0.99])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [3, 5])
    @pytest.mark.parametrize("feature_dim", [[5], [2, 5]])
    @pytest.mark.parametrize("random_gamma,rolling_gamma", [[False, None]])
    def test_td1_multi(
        self, device, gamma, N, T, random_gamma, rolling_gamma, feature_dim
    ):
        torch.manual_seed(0)

        D = feature_dim
        time_dim = -1 - len(D)
        done = torch.zeros(*N, T, *D, device=device, dtype=torch.bool)
        terminated = done.clone().bernoulli_(0.1)
        done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, *D, device=device)
        state_value = torch.randn(*N, T, *D, device=device)
        next_state_value = torch.randn(*N, T, *D, device=device)
        if random_gamma:
            gamma = torch.rand_like(reward) * gamma

        r1 = vec_td1_advantage_estimate(
            gamma,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
            time_dim=time_dim,
        )
        r2 = td1_advantage_estimate(
            gamma,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
            time_dim=time_dim,
        )
        if len(D) == 2:
            r3 = torch.cat(
                [
                    vec_td1_advantage_estimate(
                        gamma,
                        state_value[..., i : i + 1, j],
                        next_state_value[..., i : i + 1, j],
                        reward[..., i : i + 1, j],
                        done=done[..., i : i + 1, j],
                        terminated=terminated[..., i : i + 1, j],
                        rolling_gamma=rolling_gamma,
                        time_dim=-2,
                    )
                    for i in range(D[0])
                    for j in range(D[1])
                ],
                -1,
            ).unflatten(-1, D)
            r4 = torch.cat(
                [
                    td1_advantage_estimate(
                        gamma,
                        state_value[..., i : i + 1, j],
                        next_state_value[..., i : i + 1, j],
                        reward[..., i : i + 1, j],
                        done=done[..., i : i + 1, j],
                        terminated=terminated[..., i : i + 1, j],
                        rolling_gamma=rolling_gamma,
                        time_dim=-2,
                    )
                    for i in range(D[0])
                    for j in range(D[1])
                ],
                -1,
            ).unflatten(-1, D)
        else:
            r3 = torch.cat(
                [
                    vec_td1_advantage_estimate(
                        gamma,
                        state_value[..., i : i + 1],
                        next_state_value[..., i : i + 1],
                        reward[..., i : i + 1],
                        done=done[..., i : i + 1],
                        terminated=terminated[..., i : i + 1],
                        rolling_gamma=rolling_gamma,
                        time_dim=-2,
                    )
                    for i in range(D[0])
                ],
                -1,
            )
            r4 = torch.cat(
                [
                    td1_advantage_estimate(
                        gamma,
                        state_value[..., i : i + 1],
                        next_state_value[..., i : i + 1],
                        reward[..., i : i + 1],
                        done=done[..., i : i + 1],
                        terminated=terminated[..., i : i + 1],
                        rolling_gamma=rolling_gamma,
                        time_dim=-2,
                    )
                    for i in range(D[0])
                ],
                -1,
            )

        torch.testing.assert_close(r4, r2, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(r3, r1, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.99, 0.5, 0.1])
    @pytest.mark.parametrize("lmbda", [0.99, 0.5, 0.1])
    @pytest.mark.parametrize("N", [(1,), (3,), (7, 3)])
    @pytest.mark.parametrize("T", [200, 5, 3])
    @pytest.mark.parametrize("dtype", [torch.float, torch.double])
    @pytest.mark.parametrize("has_done", [False, True])
    def test_gae(self, device, gamma, lmbda, N, T, dtype, has_done):
        torch.manual_seed(0)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated = done.clone()
        if has_done:
            terminated = terminated.bernoulli_(0.1)
            done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, 1, device=device, dtype=dtype)
        state_value = torch.randn(*N, T, 1, device=device, dtype=dtype)
        next_state_value = torch.randn(*N, T, 1, device=device, dtype=dtype)

        r1 = vec_generalized_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        r2 = generalized_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )

        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("N", [(1,), (8,), (7, 3)])
    @pytest.mark.parametrize("dtype", [torch.float, torch.double])
    @pytest.mark.parametrize("has_done", [True, False])
    @pytest.mark.parametrize(
        "gamma_tensor", ["scalar", "tensor", "tensor_single_element"]
    )
    @pytest.mark.parametrize(
        "lmbda_tensor", ["scalar", "tensor", "tensor_single_element"]
    )
    def test_gae_param_as_tensor(
        self, device, N, dtype, has_done, gamma_tensor, lmbda_tensor
    ):
        torch.manual_seed(0)

        gamma = 0.95
        lmbda = 0.90
        T = 200

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated = done.clone()
        if has_done:
            terminated = terminated.bernoulli_(0.1)
            done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, 1, device=device, dtype=dtype)
        state_value = torch.randn(*N, T, 1, device=device, dtype=dtype)
        next_state_value = torch.randn(*N, T, 1, device=device, dtype=dtype)

        if gamma_tensor == "tensor":
            gamma_vec = torch.full_like(reward, gamma)
        elif gamma_tensor == "tensor_single_element":
            gamma_vec = torch.as_tensor([gamma], device=device)
        else:
            gamma_vec = gamma

        if lmbda_tensor == "tensor":
            lmbda_vec = torch.full_like(reward, lmbda)
        elif gamma_tensor == "tensor_single_element":
            lmbda_vec = torch.as_tensor([lmbda], device=device)
        else:
            lmbda_vec = lmbda

        r1 = vec_generalized_advantage_estimate(
            gamma_vec,
            lmbda_vec,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        r2 = generalized_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.99, 0.5, 0.1])
    @pytest.mark.parametrize("lmbda", [0.99, 0.5, 0.1])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [100, 3])
    @pytest.mark.parametrize("dtype", [torch.float, torch.double])
    @pytest.mark.parametrize("feature_dim", [[5], [2, 5]])
    @pytest.mark.parametrize("has_done", [True, False])
    def test_gae_multidim(
        self, device, gamma, lmbda, N, T, dtype, has_done, feature_dim
    ):
        D = feature_dim
        time_dim = -1 - len(D)

        torch.manual_seed(0)

        done = torch.zeros(*N, T, *D, device=device, dtype=torch.bool)
        terminated = done.clone()
        if has_done:
            terminated = terminated.bernoulli_(0.1)
            done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, *D, device=device, dtype=dtype)
        state_value = torch.randn(*N, T, *D, device=device, dtype=dtype)
        next_state_value = torch.randn(*N, T, *D, device=device, dtype=dtype)

        r1 = vec_generalized_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            time_dim=time_dim,
        )
        r2 = generalized_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            time_dim=time_dim,
        )
        if len(D) == 2:
            r3 = [
                vec_generalized_advantage_estimate(
                    gamma,
                    lmbda,
                    state_value[..., i : i + 1, j],
                    next_state_value[..., i : i + 1, j],
                    reward[..., i : i + 1, j],
                    done=done[..., i : i + 1, j],
                    terminated=terminated[..., i : i + 1, j],
                    time_dim=-2,
                )
                for i in range(D[0])
                for j in range(D[1])
            ]
            r4 = [
                generalized_advantage_estimate(
                    gamma,
                    lmbda,
                    state_value[..., i : i + 1, j],
                    next_state_value[..., i : i + 1, j],
                    reward[..., i : i + 1, j],
                    terminated=terminated[..., i : i + 1, j],
                    done=done[..., i : i + 1, j],
                    time_dim=-2,
                )
                for i in range(D[0])
                for j in range(D[1])
            ]
        else:
            r3 = [
                vec_generalized_advantage_estimate(
                    gamma,
                    lmbda,
                    state_value[..., i : i + 1],
                    next_state_value[..., i : i + 1],
                    reward[..., i : i + 1],
                    done=done[..., i : i + 1],
                    terminated=terminated[..., i : i + 1],
                    time_dim=-2,
                )
                for i in range(D[0])
            ]
            r4 = [
                generalized_advantage_estimate(
                    gamma,
                    lmbda,
                    state_value[..., i : i + 1],
                    next_state_value[..., i : i + 1],
                    reward[..., i : i + 1],
                    done=done[..., i : i + 1],
                    terminated=terminated[..., i : i + 1],
                    time_dim=-2,
                )
                for i in range(D[0])
            ]

        list3 = list(zip(*r3))
        list4 = list(zip(*r4))
        r3 = [torch.cat(list3[0], -1), torch.cat(list3[1], -1)]
        r4 = [torch.cat(list4[0], -1), torch.cat(list4[1], -1)]
        if len(D) == 2:
            r3 = [r3[0].unflatten(-1, D), r3[1].unflatten(-1, D)]
            r4 = [r4[0].unflatten(-1, D), r4[1].unflatten(-1, D)]
        torch.testing.assert_close(r2, r4, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(r1, r3, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.99, 0.5, 0.1])
    @pytest.mark.parametrize("N", [(1,), (3,), (7, 3)])
    @pytest.mark.parametrize("T", [200, 5, 3])
    @pytest.mark.parametrize("dtype", [torch.float, torch.double])
    @pytest.mark.parametrize("has_done", [False, True])
    def test_vtrace(self, device, gamma, N, T, dtype, has_done):
        torch.manual_seed(0)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated = done.clone()
        if has_done:
            terminated = terminated.bernoulli_(0.1)
            done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, 1, device=device, dtype=dtype)
        state_value = torch.randn(*N, T, 1, device=device, dtype=dtype)
        next_state_value = torch.randn(*N, T, 1, device=device, dtype=dtype)
        log_pi = torch.log(torch.rand(*N, T, 1, device=device, dtype=dtype))
        log_mu = torch.log(torch.rand(*N, T, 1, device=device, dtype=dtype))

        _, value_target = vtrace_advantage_estimate(
            gamma,
            log_pi,
            log_mu,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )

        assert not torch.isnan(value_target).any()
        assert not torch.isinf(value_target).any()

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.99, 0.5, 0.1])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [100, 3])
    @pytest.mark.parametrize("dtype", [torch.float, torch.double])
    @pytest.mark.parametrize("feature_dim", [[5], [2, 5]])
    @pytest.mark.parametrize("has_done", [True, False])
    def test_vtrace_multidim(self, device, gamma, N, T, dtype, has_done, feature_dim):
        D = feature_dim
        time_dim = -1 - len(D)

        torch.manual_seed(0)

        done = torch.zeros(*N, T, *D, device=device, dtype=torch.bool)
        terminated = done.clone()
        if has_done:
            terminated = terminated.bernoulli_(0.1)
            done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, *D, device=device, dtype=dtype)
        state_value = torch.randn(*N, T, *D, device=device, dtype=dtype)
        next_state_value = torch.randn(*N, T, *D, device=device, dtype=dtype)
        log_pi = torch.log(torch.rand(*N, T, *D, device=device, dtype=dtype))
        log_mu = torch.log(torch.rand(*N, T, *D, device=device, dtype=dtype))

        r1 = vtrace_advantage_estimate(
            gamma,
            log_pi,
            log_mu,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            time_dim=time_dim,
        )
        if len(D) == 2:
            r2 = [
                vtrace_advantage_estimate(
                    gamma,
                    log_pi[..., i : i + 1, j],
                    log_mu[..., i : i + 1, j],
                    state_value[..., i : i + 1, j],
                    next_state_value[..., i : i + 1, j],
                    reward[..., i : i + 1, j],
                    terminated=terminated[..., i : i + 1, j],
                    done=done[..., i : i + 1, j],
                    time_dim=-2,
                )
                for i in range(D[0])
                for j in range(D[1])
            ]
        else:
            r2 = [
                vtrace_advantage_estimate(
                    gamma,
                    log_pi[..., i : i + 1],
                    log_mu[..., i : i + 1],
                    state_value[..., i : i + 1],
                    next_state_value[..., i : i + 1],
                    reward[..., i : i + 1],
                    done=done[..., i : i + 1],
                    terminated=terminated[..., i : i + 1],
                    time_dim=-2,
                )
                for i in range(D[0])
            ]

        list2 = list(zip(*r2))
        r2 = [torch.cat(list2[0], -1), torch.cat(list2[1], -1)]
        if len(D) == 2:
            r2 = [r2[0].unflatten(-1, D), r2[1].unflatten(-1, D)]
        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.5, 0.99, 0.1])
    @pytest.mark.parametrize("lmbda", [0.1, 0.5, 0.99])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [200, 5, 3])
    @pytest.mark.parametrize("has_done", [True, False])
    def test_tdlambda_tensor_gamma(self, device, gamma, lmbda, N, T, has_done):
        """Tests vec_td_lambda_advantage_estimate against itself with
        gamma being a tensor or a scalar

        """
        torch.manual_seed(0)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        if has_done:
            terminated = terminated.bernoulli_(0.1)
            done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        gamma_tensor = torch.full((*N, T, 1), gamma, device=device)
        v1 = vec_td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        v2 = vec_td_lambda_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

        # # same with last done being true
        done[..., -1, :] = True  # terminating trajectory
        terminated[..., -1, :] = True  # terminating trajectory
        gamma_tensor[..., -1, :] = 0.0

        v1 = vec_td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        v2 = vec_td_lambda_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.5, 0.99])
    @pytest.mark.parametrize("lmbda", [0.25, 0.99])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [3, 100])
    @pytest.mark.parametrize("F", [1, 4])
    @pytest.mark.parametrize("has_done", [True, False])
    @pytest.mark.parametrize(
        "gamma_tensor", ["scalar", "tensor", "tensor_single_element"]
    )
    @pytest.mark.parametrize("lmbda_tensor", ["scalar", "tensor_single_element"])
    def test_tdlambda_tensor_gamma_single_element(
        self, device, gamma, lmbda, N, T, F, has_done, gamma_tensor, lmbda_tensor
    ):
        """Tests vec_td_lambda_advantage_estimate against itself with
        gamma being a tensor or a scalar

        """
        torch.manual_seed(0)

        done = torch.zeros(*N, T, F, device=device, dtype=torch.bool)
        terminated = torch.zeros(*N, T, F, device=device, dtype=torch.bool)
        if has_done:
            terminated = terminated.bernoulli_(0.1)
            done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, F, device=device)
        state_value = torch.randn(*N, T, F, device=device)
        next_state_value = torch.randn(*N, T, F, device=device)

        if gamma_tensor == "tensor":
            gamma_vec = torch.full_like(reward, gamma)
        elif gamma_tensor == "tensor_single_element":
            gamma_vec = torch.as_tensor([gamma], device=device)
        else:
            gamma_vec = gamma

        if gamma_tensor == "tensor_single_element":
            lmbda_vec = torch.as_tensor([lmbda], device=device)
        else:
            lmbda_vec = lmbda

        v1 = vec_td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        v2 = vec_td_lambda_advantage_estimate(
            gamma_vec,
            lmbda_vec,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

        # # same with last done being true
        done[..., -1, :] = True  # terminating trajectory
        terminated[..., -1, :] = True  # terminating trajectory

        v1 = vec_td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        v2 = vec_td_lambda_advantage_estimate(
            gamma_vec,
            lmbda_vec,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.5, 0.99, 0.1])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [3, 5, 200])
    @pytest.mark.parametrize("has_done", [True, False])
    def test_td1_tensor_gamma(self, device, gamma, N, T, has_done):
        """Tests vec_td_lambda_advantage_estimate against itself with
        gamma being a tensor or a scalar

        """
        torch.manual_seed(0)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        if has_done:
            terminated = terminated.bernoulli_(0.1)
            done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        gamma_tensor = torch.full((*N, T, 1), gamma, device=device)

        v1 = vec_td1_advantage_estimate(
            gamma,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        v2 = vec_td1_advantage_estimate(
            gamma_tensor,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

        # # same with last done being true
        done[..., -1, :] = True  # terminating trajectory
        terminated[..., -1, :] = True  # terminating trajectory
        gamma_tensor[..., -1, :] = 0.0

        v1 = vec_td1_advantage_estimate(
            gamma,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        v2 = vec_td1_advantage_estimate(
            gamma_tensor,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.5, 0.99, 0.1])
    @pytest.mark.parametrize("lmbda", [0.1, 0.5, 0.99])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [3, 5, 50])
    @pytest.mark.parametrize("has_done", [True, False])
    def test_vectdlambda_tensor_gamma(
        self, device, gamma, lmbda, N, T, dtype_fixture, has_done  # noqa
    ):
        """Tests td_lambda_advantage_estimate against vec_td_lambda_advantage_estimate
        with gamma being a tensor or a scalar

        """

        torch.manual_seed(0)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        if has_done:
            terminated = terminated.bernoulli_(0.1)
            done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        gamma_tensor = torch.full((*N, T, 1), gamma, device=device)

        v1 = td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        v2 = vec_td_lambda_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

        # same with last done being true
        done[..., -1, :] = True  # terminating trajectory
        terminated[..., -1, :] = True  # terminating trajectory
        gamma_tensor[..., -1, :] = 0.0

        v1 = td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        v2 = vec_td_lambda_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.5, 0.99, 0.1])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [3, 5, 50])
    @pytest.mark.parametrize("has_done", [True, False])
    def test_vectd1_tensor_gamma(
        self, device, gamma, N, T, dtype_fixture, has_done  # noqa
    ):
        """Tests td_lambda_advantage_estimate against vec_td_lambda_advantage_estimate
        with gamma being a tensor or a scalar

        """

        torch.manual_seed(0)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        if has_done:
            terminated = terminated.bernoulli_(0.1)
            done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        gamma_tensor = torch.full((*N, T, 1), gamma, device=device)

        v1 = td1_advantage_estimate(
            gamma,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        v2 = vec_td1_advantage_estimate(
            gamma_tensor,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

        # same with last done being true
        done[..., -1, :] = True  # terminating trajectory
        terminated[..., -1, :] = True  # terminating trajectory
        gamma_tensor[..., -1, :] = 0.0

        v1 = td1_advantage_estimate(
            gamma,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        v2 = vec_td1_advantage_estimate(
            gamma_tensor,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("lmbda", [0.1, 0.5, 0.99])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [50, 3])
    @pytest.mark.parametrize("rolling_gamma", [True, False, None])
    @pytest.mark.parametrize("has_done", [True, False])
    @pytest.mark.parametrize("seed", range(1))
    def test_vectdlambda_rand_gamma(
        self, device, lmbda, N, T, rolling_gamma, dtype_fixture, has_done, seed  # noqa
    ):
        """Tests td_lambda_advantage_estimate against vec_td_lambda_advantage_estimate
        with gamma being a random tensor

        """
        torch.manual_seed(seed)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        if has_done:
            terminated = terminated.bernoulli_(0.1)
            done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        # avoid low values of gamma
        gamma_tensor = 0.5 + torch.rand_like(next_state_value) / 2

        v1 = td_lambda_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
        )
        if rolling_gamma is False and not done[..., 1:, :][done[..., :-1, :]].all():
            # if a not-done follows a done, then rolling_gamma=False cannot be used
            with pytest.raises(
                NotImplementedError, match="When using rolling_gamma=False"
            ):
                vec_td_lambda_advantage_estimate(
                    gamma_tensor,
                    lmbda,
                    state_value,
                    next_state_value,
                    reward,
                    done=done,
                    terminated=terminated,
                    rolling_gamma=rolling_gamma,
                )
            return
        elif rolling_gamma is False:
            with pytest.raises(
                NotImplementedError, match=r"The vectorized version of TD"
            ):
                vec_td_lambda_advantage_estimate(
                    gamma_tensor,
                    lmbda,
                    state_value,
                    next_state_value,
                    reward,
                    done=done,
                    terminated=terminated,
                    rolling_gamma=rolling_gamma,
                )
            return
        v2 = vec_td_lambda_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
        )
        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [50, 3])
    @pytest.mark.parametrize("rolling_gamma", [True, False, None])
    @pytest.mark.parametrize("has_done", [True, False])
    @pytest.mark.parametrize("seed", range(1))
    def test_vectd1_rand_gamma(
        self, device, N, T, rolling_gamma, dtype_fixture, has_done, seed  # noqa
    ):
        """Tests td_lambda_advantage_estimate against vec_td_lambda_advantage_estimate
        with gamma being a random tensor

        """
        torch.manual_seed(seed)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        if has_done:
            terminated = terminated.bernoulli_(0.1)
            done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        # avoid low values of gamma
        gamma_tensor = 0.5 + torch.rand_like(next_state_value) / 2

        v1 = td1_advantage_estimate(
            gamma_tensor,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
        )
        if (
            rolling_gamma is False
            and not terminated[..., 1:, :][terminated[..., :-1, :]].all()
        ):
            # if a not-done follows a done, then rolling_gamma=False cannot be used
            with pytest.raises(
                NotImplementedError, match="When using rolling_gamma=False"
            ):
                vec_td1_advantage_estimate(
                    gamma_tensor,
                    state_value,
                    next_state_value,
                    reward,
                    done=done,
                    terminated=terminated,
                    rolling_gamma=rolling_gamma,
                )
            return
        elif rolling_gamma is False:
            with pytest.raises(
                NotImplementedError, match="The vectorized version of TD"
            ):
                vec_td1_advantage_estimate(
                    gamma_tensor,
                    state_value,
                    next_state_value,
                    reward,
                    done=done,
                    terminated=terminated,
                    rolling_gamma=rolling_gamma,
                )
            return
        v2 = vec_td1_advantage_estimate(
            gamma_tensor,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
        )
        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.99, "rand"])
    @pytest.mark.parametrize("N", [(3,), (3, 7)])
    @pytest.mark.parametrize("T", [3, 5, 200])
    @pytest.mark.parametrize("rolling_gamma", [True, False])
    def test_custom_conv1d_tensor(self, device, gamma, N, T, rolling_gamma):
        """
        Tests the _custom_conv1d logic against a manual for-loop implementation
        """
        torch.manual_seed(0)

        if gamma == "rand":
            gamma = torch.rand(*N, T, 1, device=device)
            rand_gamma = True
        else:
            gamma = torch.full((*N, T, 1), gamma, device=device)
            rand_gamma = False

        values = torch.randn(*N, 1, T, device=device)
        out = torch.zeros(*N, 1, T, device=device)
        if rand_gamma and not rolling_gamma:
            for i in range(T):
                for j in reversed(range(i, T)):
                    out[..., i] = out[..., i] * gamma[..., i, :] + values[..., j]
        else:
            prev_val = 0.0
            for i in reversed(range(T)):
                prev_val = out[..., i] = prev_val * gamma[..., i, :] + values[..., i]

        gammas = _make_gammas_tensor(gamma, T, rolling_gamma)
        gammas = gammas.cumprod(-2)
        out_custom = _custom_conv1d(values.view(-1, 1, T), gammas).reshape(values.shape)

        torch.testing.assert_close(out, out_custom, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("N", [(3,), (3, 7)])
    @pytest.mark.parametrize("T", [3, 5, 200])
    @pytest.mark.parametrize("rolling_gamma", [True, False])
    def test_successive_traj_tdlambda(self, device, N, T, rolling_gamma):
        """Tests td_lambda_advantage_estimate against vec_td_lambda_advantage_estimate
        with gamma being a random tensor

        """
        torch.manual_seed(0)

        lmbda = torch.rand([]).item()

        terminated = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated[..., T // 2 - 1, :] = 1
        done = terminated.clone()
        done[..., -1, :] = 1

        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        # avoid low values of gamma
        gamma_tensor = 0.5 + torch.rand_like(next_state_value) / 2

        v1 = td_lambda_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
        )
        v1a = td_lambda_advantage_estimate(
            gamma_tensor[..., : T // 2, :],
            lmbda,
            state_value[..., : T // 2, :],
            next_state_value[..., : T // 2, :],
            reward[..., : T // 2, :],
            done=done[..., : T // 2, :],
            terminated=terminated[..., : T // 2, :],
            rolling_gamma=rolling_gamma,
        )
        v1b = td_lambda_advantage_estimate(
            gamma_tensor[..., T // 2 :, :],
            lmbda,
            state_value[..., T // 2 :, :],
            next_state_value[..., T // 2 :, :],
            reward[..., T // 2 :, :],
            done=done[..., T // 2 :, :],
            terminated=terminated[..., T // 2 :, :],
            rolling_gamma=rolling_gamma,
        )
        torch.testing.assert_close(v1, torch.cat([v1a, v1b], -2), rtol=1e-4, atol=1e-4)

        if not rolling_gamma:
            with pytest.raises(
                NotImplementedError, match="When using rolling_gamma=False"
            ):
                vec_td_lambda_advantage_estimate(
                    gamma_tensor,
                    lmbda,
                    state_value,
                    next_state_value,
                    reward,
                    done=done,
                    terminated=terminated,
                    rolling_gamma=rolling_gamma,
                )
            return
        v2 = vec_td_lambda_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
        )
        v2a = vec_td_lambda_advantage_estimate(
            gamma_tensor[..., : T // 2, :],
            lmbda,
            state_value[..., : T // 2, :],
            next_state_value[..., : T // 2, :],
            reward[..., : T // 2, :],
            done=done[..., : T // 2, :],
            terminated=terminated[..., : T // 2, :],
            rolling_gamma=rolling_gamma,
        )
        v2b = vec_td_lambda_advantage_estimate(
            gamma_tensor[..., T // 2 :, :],
            lmbda,
            state_value[..., T // 2 :, :],
            next_state_value[..., T // 2 :, :],
            reward[..., T // 2 :, :],
            done=done[..., T // 2 :, :],
            terminated=terminated[..., T // 2 :, :],
            rolling_gamma=rolling_gamma,
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(v1a, v2a, rtol=1e-4, atol=1e-4)

        torch.testing.assert_close(v1b, v2b, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(v2, torch.cat([v2a, v2b], -2), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("N", [(3,), (3, 7)])
    @pytest.mark.parametrize("T", [3, 5, 200])
    def test_successive_traj_tdadv(self, device, N, T):
        """Tests td_lambda_advantage_estimate against vec_td_lambda_advantage_estimate
        with gamma being a random tensor

        """
        torch.manual_seed(0)

        # for td0, a done that is not terminated has no effect
        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        done[..., T // 2 - 1, :] = 1
        terminated = done.clone()

        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        # avoid low values of gamma
        gamma_tensor = 0.5 + torch.rand_like(next_state_value) / 2

        v1 = td0_advantage_estimate(
            gamma_tensor,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        v1a = td0_advantage_estimate(
            gamma_tensor[..., : T // 2, :],
            state_value[..., : T // 2, :],
            next_state_value[..., : T // 2, :],
            reward[..., : T // 2, :],
            done=done[..., : T // 2, :],
            terminated=terminated[..., : T // 2, :],
        )
        v1b = td0_advantage_estimate(
            gamma_tensor[..., T // 2 :, :],
            state_value[..., T // 2 :, :],
            next_state_value[..., T // 2 :, :],
            reward[..., T // 2 :, :],
            done=done[..., T // 2 :, :],
            terminated=terminated[..., T // 2 :, :],
        )
        torch.testing.assert_close(v1, torch.cat([v1a, v1b], -2), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    def test_args_kwargs_timedim(self, device):
        torch.manual_seed(0)

        lmbda = 0.95
        N = (2, 3)
        B = (4,)
        T = 20

        terminated = torch.zeros(*N, T, *B, 1, device=device, dtype=torch.bool)
        terminated[..., T // 2 - 1, :, :] = 1
        done = terminated.clone()
        done[..., -1, :, :] = 1

        reward = torch.randn(*N, T, *B, 1, device=device)
        state_value = torch.randn(*N, T, *B, 1, device=device)
        next_state_value = torch.randn(*N, T, *B, 1, device=device)

        # avoid low values of gamma
        gamma = 0.95

        v1 = vec_generalized_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            time_dim=-3,
        )[0]

        v2 = vec_generalized_advantage_estimate(
            gamma=gamma,
            lmbda=lmbda,
            state_value=state_value,
            next_state_value=next_state_value,
            reward=reward,
            done=done,
            terminated=terminated,
            time_dim=-3,
        )[0]

        with pytest.raises(TypeError, match="positional arguments"):
            v3 = vec_generalized_advantage_estimate(
                gamma,
                lmbda,
                state_value,
                next_state_value,
                reward,
                done,
                terminated,
                -3,
            )[0]

        v3 = vec_generalized_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done,
            terminated,
            time_dim=-3,
        )[0]

        v4 = vec_generalized_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done,
            terminated,
            time_dim=2,
        )[0]

        v5 = vec_generalized_advantage_estimate(
            gamma=gamma,
            lmbda=lmbda,
            state_value=state_value,
            next_state_value=next_state_value,
            reward=reward,
            done=done,
            terminated=terminated,
            time_dim=-3,
        )[0]
        torch.testing.assert_close(v1, v2)
        torch.testing.assert_close(v1, v3)
        torch.testing.assert_close(v1, v4)
        torch.testing.assert_close(v1, v5)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("N", [(3,), (3, 7)])
    @pytest.mark.parametrize("T", [3, 5, 200])
    def test_successive_traj_gae(
        self,
        device,
        N,
        T,
    ):
        """Tests td_lambda_advantage_estimate against vec_td_lambda_advantage_estimate
        with gamma being a random tensor

        """
        torch.manual_seed(0)

        lmbda = torch.rand([]).item()

        terminated = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated[..., T // 2 - 1, :] = 1
        done = terminated.clone()
        done[..., -1, :] = 1

        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        # avoid low values of gamma
        gamma_tensor = 0.95

        v1 = generalized_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )[0]
        v1a = generalized_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value[..., : T // 2, :],
            next_state_value[..., : T // 2, :],
            reward[..., : T // 2, :],
            done=done[..., : T // 2, :],
            terminated=terminated[..., : T // 2, :],
        )[0]
        v1b = generalized_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value[..., T // 2 :, :],
            next_state_value[..., T // 2 :, :],
            reward[..., T // 2 :, :],
            done=done[..., T // 2 :, :],
            terminated=terminated[..., T // 2 :, :],
        )[0]
        torch.testing.assert_close(v1, torch.cat([v1a, v1b], -2), rtol=1e-4, atol=1e-4)

        v2 = vec_generalized_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )[0]
        v2a = vec_generalized_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value[..., : T // 2, :],
            next_state_value[..., : T // 2, :],
            reward[..., : T // 2, :],
            done=done[..., : T // 2, :],
            terminated=terminated[..., : T // 2, :],
        )[0]
        v2b = vec_generalized_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value[..., T // 2 :, :],
            next_state_value[..., T // 2 :, :],
            reward[..., T // 2 :, :],
            done=done[..., T // 2 :, :],
            terminated=terminated[..., T // 2 :, :],
        )[0]
        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(v2, torch.cat([v2a, v2b], -2), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not _has_functorch,
    reason=f"no vmap allowed without functorch, error: {FUNCTORCH_ERR}",
)
@pytest.mark.parametrize(
    "dest,expected_dtype,expected_device",
    list(
        zip(
            get_available_devices(),
            [torch.float] * len(get_available_devices()),
            get_available_devices(),
        )
    )
    + [
        ["cuda", torch.float, "cuda:0"],
        ["double", torch.double, "cpu"],
        [torch.double, torch.double, "cpu"],
        [torch.half, torch.half, "cpu"],
        ["half", torch.half, "cpu"],
    ],
)
@set_composite_lp_aggregate(False)
def test_shared_params(dest, expected_dtype, expected_device):
    if torch.cuda.device_count() == 0 and dest == "cuda":
        pytest.skip("no cuda device available")
    module_hidden = torch.nn.Linear(4, 4)
    td_module_hidden = TensorDictModule(
        module=module_hidden,
        in_keys=["observation"],
        out_keys=["hidden"],
    )
    module_action = TensorDictModule(
        nn.Sequential(nn.Linear(4, 8), NormalParamExtractor()),
        in_keys=["hidden"],
        out_keys=["loc", "scale"],
    )
    td_module_action = ProbabilisticActor(
        module=module_action,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        spec=None,
        distribution_class=TanhNormal,
        return_log_prob=True,
    )
    module_value = torch.nn.Linear(4, 1)
    td_module_value = ValueOperator(module=module_value, in_keys=["hidden"])
    td_module = ActorValueOperator(td_module_hidden, td_module_action, td_module_value)

    class MyLoss(LossModule):
        actor_network: TensorDictModule
        actor_network_params: TensorDict
        target_actor_network_params: TensorDict
        qvalue_network: TensorDictModule
        qvalue_network_params: TensorDict
        target_qvalue_network_params: TensorDict

        def __init__(self, actor_network, qvalue_network):
            super().__init__()
            self.convert_to_functional(
                actor_network,
                "actor_network",
                create_target_params=True,
            )
            self.convert_to_functional(
                qvalue_network,
                "qvalue_network",
                3,
                create_target_params=True,
                compare_against=list(actor_network.parameters()),
            )

        def _forward_value_estimator_keys(self, **kwargs) -> None:
            pass

    actor_network = td_module.get_policy_operator()
    value_network = td_module.get_value_operator()

    loss = MyLoss(actor_network, value_network)
    # modify params
    for p in loss.parameters():
        if p.requires_grad:
            p.data += torch.randn_like(p)

    assert len([p for p in loss.parameters() if p.requires_grad]) == 6
    assert (
        len(loss.actor_network_params.keys(include_nested=True, leaves_only=True)) == 4
    )
    assert (
        len(loss.qvalue_network_params.keys(include_nested=True, leaves_only=True)) == 4
    )
    for p in loss.actor_network_params.values(include_nested=True, leaves_only=True):
        assert isinstance(p, nn.Parameter) or isinstance(p, Buffer)
    for i, (key, value) in enumerate(
        loss.qvalue_network_params.items(include_nested=True, leaves_only=True)
    ):
        p1 = value
        p2 = loss.actor_network_params[key]
        assert (p1 == p2).all()
        if i == 1:
            break

    # map module
    if dest == "double":
        loss = loss.double()
    elif dest == "cuda":
        loss = loss.cuda()
    elif dest == "half":
        loss = loss.half()
    else:
        loss = loss.to(dest)

    for p in loss.actor_network_params.values(include_nested=True, leaves_only=True):
        assert isinstance(p, nn.Parameter)
        assert p.dtype is expected_dtype
        assert p.device == torch.device(expected_device)
    for i, (key, qvalparam) in enumerate(
        loss.qvalue_network_params.items(include_nested=True, leaves_only=True)
    ):
        assert qvalparam.dtype is expected_dtype, (key, qvalparam)
        assert qvalparam.device == torch.device(expected_device), key
        assert (qvalparam == loss.actor_network_params[key]).all(), key
        if i == 1:
            break


class TestAdv:
    @pytest.fixture(scope="class", autouse=True)
    def _composite_log_prob(self):
        setter = set_composite_lp_aggregate(False)
        setter.set()
        yield
        setter.unset()

    @pytest.mark.parametrize(
        "adv,kwargs",
        [
            [GAE, {"lmbda": 0.95}],
            [TD1Estimator, {}],
            [TDLambdaEstimator, {"lmbda": 0.95}],
            [VTrace, {}],
        ],
    )
    def test_dispatch(
        self,
        adv,
        kwargs,
    ):
        value_net = TensorDictModule(
            nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
        )
        if adv is VTrace:
            actor_net = TensorDictModule(
                nn.Linear(3, 4), in_keys=["obs"], out_keys=["logits"]
            )
            actor_net = ProbabilisticActor(
                module=actor_net,
                in_keys=["logits"],
                out_keys=["action"],
                distribution_class=OneHotCategorical,
                return_log_prob=True,
            )
            module = adv(
                gamma=0.98,
                actor_network=actor_net,
                value_network=value_net,
                differentiable=False,
                **kwargs,
            )
            kwargs = {
                "obs": torch.randn(1, 10, 3),
                "action_log_prob": torch.log(torch.rand(1, 10, 1)),
                "next_reward": torch.randn(1, 10, 1, requires_grad=True),
                "next_done": torch.zeros(1, 10, 1, dtype=torch.bool),
                "next_terminated": torch.zeros(1, 10, 1, dtype=torch.bool),
                "next_obs": torch.randn(1, 10, 3),
            }
        else:
            module = adv(
                gamma=0.98,
                value_network=value_net,
                differentiable=False,
                **kwargs,
            )
            kwargs = {
                "obs": torch.randn(1, 10, 3),
                "next_reward": torch.randn(1, 10, 1, requires_grad=True),
                "next_done": torch.zeros(1, 10, 1, dtype=torch.bool),
                "next_terminated": torch.zeros(1, 10, 1, dtype=torch.bool),
                "next_obs": torch.randn(1, 10, 3),
            }
        advantage, value_target = module(**kwargs)
        assert advantage.shape == torch.Size([1, 10, 1])
        assert value_target.shape == torch.Size([1, 10, 1])

    @pytest.mark.parametrize(
        "adv,kwargs",
        [
            [GAE, {"lmbda": 0.95}],
            [TD1Estimator, {}],
            [TDLambdaEstimator, {"lmbda": 0.95}],
            [VTrace, {}],
        ],
    )
    def test_diff_reward(
        self,
        adv,
        kwargs,
    ):
        value_net = TensorDictModule(
            nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
        )
        if adv is VTrace:
            actor_net = TensorDictModule(
                nn.Linear(3, 4), in_keys=["obs"], out_keys=["logits"]
            )
            actor_net = ProbabilisticActor(
                module=actor_net,
                in_keys=["logits"],
                out_keys=["action"],
                distribution_class=OneHotCategorical,
                return_log_prob=True,
            )
            module = adv(
                gamma=0.98,
                actor_network=actor_net,
                value_network=value_net,
                differentiable=True,
                **kwargs,
            )
            td = TensorDict(
                {
                    "obs": torch.randn(1, 10, 3),
                    "action_log_prob": torch.log(torch.rand(1, 10, 1)),
                    "next": {
                        "obs": torch.randn(1, 10, 3),
                        "reward": torch.randn(1, 10, 1, requires_grad=True),
                        "done": torch.zeros(1, 10, 1, dtype=torch.bool),
                        "terminated": torch.zeros(1, 10, 1, dtype=torch.bool),
                    },
                },
                [1, 10],
            )
        else:
            module = adv(
                gamma=0.98,
                value_network=value_net,
                differentiable=True,
                **kwargs,
            )
            td = TensorDict(
                {
                    "obs": torch.randn(1, 10, 3),
                    "next": {
                        "obs": torch.randn(1, 10, 3),
                        "reward": torch.randn(1, 10, 1, requires_grad=True),
                        "done": torch.zeros(1, 10, 1, dtype=torch.bool),
                    },
                },
                [1, 10],
            )
        td = module(td.clone(False))
        # check that the advantage can't backprop to the value params
        td["advantage"].sum().backward()
        for p in value_net.parameters():
            assert p.grad is None or (p.grad == 0).all()
        # check that rewards have a grad
        assert td["next", "reward"].grad.norm() > 0

    @pytest.mark.parametrize(
        "adv,kwargs",
        [
            [GAE, {"lmbda": 0.95}],
            [TD1Estimator, {}],
            [TDLambdaEstimator, {"lmbda": 0.95}],
            [VTrace, {}],
        ],
    )
    @pytest.mark.parametrize("shifted", [True, False])
    def test_non_differentiable(self, adv, shifted, kwargs):
        value_net = TensorDictModule(
            nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
        )

        if adv is VTrace:
            actor_net = TensorDictModule(
                nn.Linear(3, 4), in_keys=["obs"], out_keys=["logits"]
            )
            actor_net = ProbabilisticActor(
                module=actor_net,
                in_keys=["logits"],
                out_keys=["action"],
                distribution_class=OneHotCategorical,
                return_log_prob=True,
            )
            module = adv(
                gamma=0.98,
                actor_network=actor_net,
                value_network=value_net,
                differentiable=False,
                shifted=shifted,
                **kwargs,
            )
            td = TensorDict(
                {
                    "obs": torch.randn(1, 10, 3),
                    "action_log_prob": torch.log(torch.rand(1, 10, 1)),
                    "next": {
                        "obs": torch.randn(1, 10, 3),
                        "reward": torch.randn(1, 10, 1, requires_grad=True),
                        "done": torch.zeros(1, 10, 1, dtype=torch.bool),
                        "terminated": torch.zeros(1, 10, 1, dtype=torch.bool),
                    },
                },
                [1, 10],
                names=[None, "time"],
            )
        else:
            module = adv(
                gamma=0.98,
                value_network=value_net,
                differentiable=False,
                shifted=shifted,
                **kwargs,
            )
            td = TensorDict(
                {
                    "obs": torch.randn(1, 10, 3),
                    "next": {
                        "obs": torch.randn(1, 10, 3),
                        "reward": torch.randn(1, 10, 1, requires_grad=True),
                        "done": torch.zeros(1, 10, 1, dtype=torch.bool),
                    },
                },
                [1, 10],
                names=[None, "time"],
            )
        td = module(td.clone(False))
        assert td["advantage"].is_leaf

    @pytest.mark.parametrize(
        "adv,kwargs",
        [
            [GAE, {"lmbda": 0.95}],
            [TD1Estimator, {}],
            [TDLambdaEstimator, {"lmbda": 0.95}],
            [VTrace, {}],
        ],
    )
    def test_time_dim(self, adv, kwargs, shifted=True):
        value_net = TensorDictModule(
            nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
        )

        if adv is VTrace:
            actor_net = TensorDictModule(
                nn.Linear(3, 4), in_keys=["obs"], out_keys=["logits"]
            )
            actor_net = ProbabilisticActor(
                module=actor_net,
                in_keys=["logits"],
                out_keys=["action"],
                distribution_class=OneHotCategorical,
                return_log_prob=True,
            )
            module_make = functools.partial(
                adv,
                gamma=0.98,
                actor_network=actor_net,
                value_network=value_net,
                differentiable=False,
                shifted=shifted,
                **kwargs,
            )
            td = TensorDict(
                {
                    "obs": torch.randn(1, 10, 3),
                    "action_log_prob": torch.log(torch.rand(1, 10, 1)),
                    "next": {
                        "obs": torch.randn(1, 10, 3),
                        "reward": torch.randn(1, 10, 1, requires_grad=True),
                        "done": torch.zeros(1, 10, 1, dtype=torch.bool),
                        "terminated": torch.zeros(1, 10, 1, dtype=torch.bool),
                    },
                },
                [1, 10],
                names=[None, "time"],
            )
        else:
            module_make = functools.partial(
                adv,
                gamma=0.98,
                value_network=value_net,
                differentiable=False,
                shifted=shifted,
                **kwargs,
            )
            td = TensorDict(
                {
                    "obs": torch.randn(1, 10, 3),
                    "next": {
                        "obs": torch.randn(1, 10, 3),
                        "reward": torch.randn(1, 10, 1, requires_grad=True),
                        "done": torch.zeros(1, 10, 1, dtype=torch.bool),
                    },
                },
                [1, 10],
                names=[None, "time"],
            )

        module_none = module_make(time_dim=None)
        module_0 = module_make(time_dim=0)
        module_1 = module_make(time_dim=1)

        td_none = module_none(td.clone(False))
        td_1 = module_1(td.clone(False))
        td_0 = module_0(td.transpose(0, 1).clone(False))
        assert_allclose_td(td_none, td_1)
        assert_allclose_td(td_none, td_0.transpose(0, 1))

        if adv is not VTrace:
            vt = module_none.value_estimate(td.clone(False))
            vt_patch = module_0.value_estimate(td.clone(False), time_dim=1)
            vt_patch2 = module_0.value_estimate(td.clone(False), time_dim=-1)
            torch.testing.assert_close(vt, vt_patch)

    @pytest.mark.parametrize(
        "adv,kwargs",
        [
            [GAE, {"lmbda": 0.95}],
            [TD1Estimator, {}],
            [TDLambdaEstimator, {"lmbda": 0.95}],
            [VTrace, {}],
        ],
    )
    @pytest.mark.parametrize("has_value_net", [True, False])
    @pytest.mark.parametrize("skip_existing", [True, False, None])
    @pytest.mark.parametrize("shifted", [True, False])
    def test_skip_existing(
        self,
        adv,
        kwargs,
        has_value_net,
        skip_existing,
        shifted,
    ):
        if has_value_net:
            value_net = TensorDictModule(
                lambda x: torch.zeros(*x.shape[:-1], 1),
                in_keys=["obs"],
                out_keys=["state_value"],
            )
        else:
            value_net = None

        if adv is VTrace:
            actor_net = TensorDictModule(
                nn.Linear(3, 4), in_keys=["obs"], out_keys=["logits"]
            )
            actor_net = ProbabilisticActor(
                module=actor_net,
                in_keys=["logits"],
                out_keys=["action"],
                distribution_class=OneHotCategorical,
                return_log_prob=True,
            )
            module = adv(
                gamma=0.98,
                actor_network=actor_net,
                value_network=value_net,
                differentiable=True,
                shifted=shifted,
                skip_existing=skip_existing,
                **kwargs,
            )
            td = TensorDict(
                {
                    "obs": torch.randn(1, 10, 3),
                    "action_log_prob": torch.log(torch.rand(1, 10, 1)),
                    "state_value": torch.ones(1, 10, 1),
                    "next": {
                        "obs": torch.randn(1, 10, 3),
                        "state_value": torch.ones(1, 10, 1),
                        "reward": torch.randn(1, 10, 1, requires_grad=True),
                        "done": torch.zeros(1, 10, 1, dtype=torch.bool),
                        "terminated": torch.zeros(1, 10, 1, dtype=torch.bool),
                    },
                },
                [1, 10],
                names=[None, "time"],
            )
        else:
            module = adv(
                gamma=0.98,
                value_network=value_net,
                differentiable=True,
                shifted=shifted,
                skip_existing=skip_existing,
                **kwargs,
            )
            td = TensorDict(
                {
                    "obs": torch.randn(1, 10, 3),
                    "state_value": torch.ones(1, 10, 1),
                    "next": {
                        "obs": torch.randn(1, 10, 3),
                        "state_value": torch.ones(1, 10, 1),
                        "reward": torch.randn(1, 10, 1, requires_grad=True),
                        "done": torch.zeros(1, 10, 1, dtype=torch.bool),
                    },
                },
                [1, 10],
                names=[None, "time"],
            )
        td = module(td.clone(False))
        if has_value_net and not skip_existing:
            exp_val = 0
        elif has_value_net and skip_existing:
            exp_val = 1
        elif not has_value_net:
            exp_val = 1
        assert (td["state_value"] == exp_val).all()
        # assert (td["next", "state_value"] == exp_val).all()

    @pytest.mark.parametrize("value", ["state_value", "state_value_test"])
    @pytest.mark.parametrize(
        "adv,kwargs",
        [
            [GAE, {"lmbda": 0.95}],
            [TD1Estimator, {}],
            [TDLambdaEstimator, {"lmbda": 0.95}],
            [VTrace, {}],
        ],
    )
    def test_set_keys(self, value, adv, kwargs):
        value_net = TensorDictModule(nn.Linear(3, 1), in_keys=["obs"], out_keys=[value])
        if adv is VTrace:
            actor_net = TensorDictModule(
                nn.Linear(3, 4), in_keys=["obs"], out_keys=["logits"]
            )
            actor_net = ProbabilisticActor(
                module=actor_net,
                in_keys=["logits"],
                out_keys=["action"],
                distribution_class=OneHotCategorical,
                return_log_prob=True,
            )
            module = adv(
                gamma=0.98,
                actor_network=actor_net,
                value_network=value_net,
                **kwargs,
            )
        else:
            module = adv(
                gamma=0.98,
                value_network=value_net,
                **kwargs,
            )
        module.set_keys(value=value)
        assert module.tensor_keys.value == value

        with pytest.raises(KeyError) as excinfo:
            module.set_keys(unknown_key="unknown_value")
            assert "unknown_value not found" in str(excinfo.value)

    @pytest.mark.parametrize(
        "adv,kwargs",
        [
            [GAE, {"lmbda": 0.95}],
            [TD1Estimator, {}],
            [TDLambdaEstimator, {"lmbda": 0.95}],
            [VTrace, {}],
        ],
    )
    def test_set_deprecated_keys(self, adv, kwargs):
        value_net = TensorDictModule(
            nn.Linear(3, 1), in_keys=["obs"], out_keys=["test_value"]
        )

        with pytest.raises(RuntimeError, match="via constructor is deprecated"):
            if adv is VTrace:
                actor_net = TensorDictModule(
                    nn.Linear(3, 4), in_keys=["obs"], out_keys=["logits"]
                )
                actor_net = ProbabilisticActor(
                    module=actor_net,
                    in_keys=["logits"],
                    out_keys=["action"],
                    distribution_class=OneHotCategorical,
                    return_log_prob=True,
                )
                module = adv(
                    gamma=0.98,
                    actor_network=actor_net,
                    value_network=value_net,
                    value_key="test_value",
                    advantage_key="advantage_test",
                    value_target_key="value_target_test",
                    **kwargs,
                )
            else:
                module = adv(
                    gamma=0.98,
                    value_network=value_net,
                    value_key="test_value",
                    advantage_key="advantage_test",
                    value_target_key="value_target_test",
                    **kwargs,
                )
            assert module.tensor_keys.value == "test_value"
            assert module.tensor_keys.advantage == "advantage_test"
            assert module.tensor_keys.value_target == "value_target_test"


class TestBase:
    @pytest.fixture(scope="class", autouse=True)
    def _composite_log_prob(self):
        setter = set_composite_lp_aggregate(False)
        setter.set()
        yield
        setter.unset()

    def test_decorators(self):
        class MyLoss(LossModule):
            def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
                assert recurrent_mode()
                assert exploration_type() is ExplorationType.DETERMINISTIC
                return TensorDict()

            def actor_loss(self, tensordict: TensorDictBase) -> TensorDictBase:
                assert recurrent_mode()
                assert exploration_type() is ExplorationType.DETERMINISTIC
                return TensorDict()

            def something_loss(self, tensordict: TensorDictBase) -> TensorDictBase:
                assert recurrent_mode()
                assert exploration_type() is ExplorationType.DETERMINISTIC
                return TensorDict()

        loss = MyLoss()
        loss.forward(None)
        loss.actor_loss(None)
        loss.something_loss(None)
        assert not recurrent_mode()

    @pytest.mark.parametrize("expand_dim", [None, 2])
    @pytest.mark.parametrize("compare_against", [True, False])
    @pytest.mark.skipif(not _has_functorch, reason="functorch is needed for expansion")
    def test_convert_to_func(self, compare_against, expand_dim):
        class MyLoss(LossModule):
            module_a: TensorDictModule
            module_b: TensorDictModule
            module_a_params: TensorDict
            module_b_params: TensorDict
            target_module_a_params: TensorDict
            target_module_b_params: TensorDict

            def __init__(self, compare_against, expand_dim):
                super().__init__()
                module1 = nn.Linear(3, 4)
                module2 = nn.Linear(3, 4)
                module3 = nn.Linear(3, 4)
                module_a = TensorDictModule(
                    nn.Sequential(module1, module2), in_keys=["a"], out_keys=["c"]
                )
                module_b = TensorDictModule(
                    nn.Sequential(module1, module3), in_keys=["b"], out_keys=["c"]
                )
                self.convert_to_functional(module_a, "module_a")
                self.convert_to_functional(
                    module_b,
                    "module_b",
                    compare_against=module_a.parameters() if compare_against else [],
                    expand_dim=expand_dim,
                )

        loss_module = MyLoss(compare_against=compare_against, expand_dim=expand_dim)

        for key in ["module.0.bias", "module.0.weight"]:
            if compare_against:
                assert not loss_module.module_b_params.flatten_keys()[key].requires_grad
            else:
                assert loss_module.module_b_params.flatten_keys()[key].requires_grad
            if expand_dim:
                assert (
                    loss_module.module_b_params.flatten_keys()[key].shape[0]
                    == expand_dim
                )
            else:
                assert (
                    loss_module.module_b_params.flatten_keys()[key].shape[0]
                    != expand_dim
                )

        for key in ["module.1.bias", "module.1.weight"]:
            loss_module.module_b_params.flatten_keys()[key].requires_grad

    def test_init_params(self):
        class MyLoss(LossModule):
            module_a: TensorDictModule
            module_b: TensorDictModule
            module_a_params: TensorDict
            module_b_params: TensorDict
            target_module_a_params: TensorDict
            target_module_b_params: TensorDict

            def __init__(self, expand_dim=2):
                super().__init__()
                module1 = nn.Linear(3, 4)
                module2 = nn.Linear(3, 4)
                module3 = nn.Linear(3, 4)
                module_a = TensorDictModule(
                    nn.Sequential(module1, module2), in_keys=["a"], out_keys=["c"]
                )
                module_b = TensorDictModule(
                    nn.Sequential(module1, module3), in_keys=["b"], out_keys=["c"]
                )
                self.convert_to_functional(module_a, "module_a")
                self.convert_to_functional(
                    module_b,
                    "module_b",
                    compare_against=module_a.parameters(),
                    expand_dim=expand_dim,
                )

        loss = MyLoss()

        module_a = loss.get_stateful_net("module_a", copy=False)
        assert module_a is loss.module_a

        module_a = loss.get_stateful_net("module_a")
        assert module_a is not loss.module_a

        def init(mod):
            if hasattr(mod, "weight"):
                mod.weight.data.zero_()
            if hasattr(mod, "bias"):
                mod.bias.data.zero_()

        module_a.apply(init)
        assert (loss.module_a_params == 0).all()

        def init(mod):
            if hasattr(mod, "weight"):
                mod.weight = torch.nn.Parameter(mod.weight.data + 1)
            if hasattr(mod, "bias"):
                mod.bias = torch.nn.Parameter(mod.bias.data + 1)

        module_a.apply(init)
        assert (loss.module_a_params == 0).all()
        loss.from_stateful_net("module_a", module_a)
        assert (loss.module_a_params == 1).all()

    def test_from_module_list(self):
        class MyLoss(LossModule):
            module_a: TensorDictModule
            module_b: TensorDictModule

            module_a_params: TensorDict
            module_b_params: TensorDict

            target_module_a_params: TensorDict
            target_module_b_params: TensorDict

            def __init__(self, module_a, module_b0, module_b1, expand_dim=2):
                super().__init__()
                self.convert_to_functional(module_a, "module_a")
                self.convert_to_functional(
                    [module_b0, module_b1],
                    "module_b",
                    # This will be ignored
                    compare_against=module_a.parameters(),
                    expand_dim=expand_dim,
                )

        module1 = nn.Linear(3, 4)
        module2 = nn.Linear(3, 4)
        module3a = nn.Linear(3, 4)
        module3b = nn.Linear(3, 4)

        module_a = TensorDictModule(
            nn.Sequential(module1, module2), in_keys=["a"], out_keys=["c"]
        )

        module_b0 = TensorDictModule(
            nn.Sequential(module1, module3a), in_keys=["b"], out_keys=["c"]
        )
        module_b1 = TensorDictModule(
            nn.Sequential(module1, module3b), in_keys=["b"], out_keys=["c"]
        )

        loss = MyLoss(module_a, module_b0, module_b1)

        # This should be extended
        assert not isinstance(
            loss.module_b_params["module", "0", "weight"], nn.Parameter
        )
        assert loss.module_b_params["module", "0", "weight"].shape[0] == 2
        assert (
            loss.module_b_params["module", "0", "weight"].data.data_ptr()
            == loss.module_a_params["module", "0", "weight"].data.data_ptr()
        )
        assert isinstance(loss.module_b_params["module", "1", "weight"], nn.Parameter)
        assert loss.module_b_params["module", "1", "weight"].shape[0] == 2
        assert (
            loss.module_b_params["module", "1", "weight"].data.data_ptr()
            != loss.module_a_params["module", "1", "weight"].data.data_ptr()
        )

    def test_tensordict_keys(self):
        """Test configurable tensordict key behavior with derived classes."""

        class MyLoss(LossModule):
            def __init__(self):
                super().__init__()

        loss_module = MyLoss()
        with pytest.raises(AttributeError):
            loss_module.set_keys()

        class MyLoss2(MyLoss):
            def _forward_value_estimator_keys(self, **kwargs) -> None:
                pass

        loss_module = MyLoss2()
        assert loss_module.set_keys() is None
        with pytest.raises(ValueError):
            loss_module.set_keys(some_key="test")

        class MyLoss3(MyLoss2):
            @dataclass
            class _AcceptedKeys:
                some_key: str = "some_value"

        loss_module = MyLoss3()
        assert loss_module.tensor_keys.some_key == "some_value"
        loss_module.set_keys(some_key="test")
        assert loss_module.tensor_keys.some_key == "test"


class TestUtils:
    @pytest.fixture(scope="class", autouse=True)
    def _composite_log_prob(self):
        setter = set_composite_lp_aggregate(False)
        setter.set()
        yield
        setter.unset()

    def test_add_random_module(self):
        class MyMod(nn.Module):
            ...

        add_random_module(MyMod)
        import torchrl.objectives.utils

        assert MyMod in torchrl.objectives.utils.RANDOM_MODULE_LIST

    def test_standardization(self):
        t = torch.arange(3 * 4 * 5 * 6, dtype=torch.float32).view(3, 4, 5, 6)
        std_t0 = _standardize(t, exclude_dims=(1, 3))
        std_t1 = (t - t.mean((0, 2), keepdim=True)) / t.std((0, 2), keepdim=True).clamp(
            1 - 6
        )
        torch.testing.assert_close(std_t0, std_t1)
        std_t = _standardize(t, (), -1, 2)
        torch.testing.assert_close(std_t, (t + 1) / 2)
        std_t = _standardize(t, ())
        torch.testing.assert_close(std_t, (t - t.mean()) / t.std())

    @pytest.mark.parametrize("B", [None, (1, ), (4, ), (2, 2, ), (1, 2, 8, )])  # fmt: skip
    @pytest.mark.parametrize("T", [1, 10])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_get_num_per_traj_no_stops(self, B, T, device):
        """check _get_num_per_traj when input contains no stops"""
        size = (*B, T) if B else (T,)

        done = torch.zeros(*size, dtype=torch.bool, device=device)
        splits = _get_num_per_traj(done)

        count = functools.reduce(operator.mul, B, 1) if B else 1
        res = torch.full((count,), T, device=device)

        torch.testing.assert_close(splits, res)

    @pytest.mark.parametrize("B", [(1, ), (3, ), (2, 2, ), (1, 2, 8, )])  # fmt: skip
    @pytest.mark.parametrize("T", [5, 100])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_get_num_per_traj(self, B, T, device):
        """check _get_num_per_traj where input contains a stop at half of each trace"""
        size = (*B, T)

        done = torch.zeros(*size, dtype=torch.bool, device=device)
        done[..., T // 2] = True
        splits = _get_num_per_traj(done)

        count = functools.reduce(operator.mul, B, 1)
        res = [T - (T + 1) // 2 + 1, (T + 1) // 2 - 1] * count
        res = torch.as_tensor(res, device=device)

        torch.testing.assert_close(splits, res)

    @pytest.mark.parametrize("B", [(1, ), (3, ), (2, 2, ), (1, 2, 8, )])  # fmt: skip
    @pytest.mark.parametrize("T", [5, 100])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_split_pad_reverse(self, B, T, device):
        """calls _split_and_pad_sequence and reverts it"""
        torch.manual_seed(42)

        size = (*B, T)
        traj = torch.rand(*size, device=device)
        done = torch.zeros(*size, dtype=torch.bool, device=device).bernoulli(0.2)
        splits = _get_num_per_traj(done)

        splitted = _split_and_pad_sequence(traj, splits)
        reversed = _inv_pad_sequence(splitted, splits).reshape(traj.shape)

        torch.testing.assert_close(traj, reversed)

    @pytest.mark.parametrize("B", [(1, ), (3, ), (2, 2, ), (1, 2, 8, )])  # fmt: skip
    @pytest.mark.parametrize("T", [5, 100])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_split_pad_no_stops(self, B, T, device):
        """_split_and_pad_sequence on trajectories without stops should not change input but flatten it along batch dimension"""
        size = (*B, T)
        count = functools.reduce(operator.mul, size, 1)

        traj = torch.arange(0, count, device=device).reshape(size)
        done = torch.zeros(*size, dtype=torch.bool, device=device)

        splits = _get_num_per_traj(done)
        splitted = _split_and_pad_sequence(traj, splits)

        traj_flat = traj.flatten(0, -2)
        torch.testing.assert_close(traj_flat, splitted)

    @pytest.mark.parametrize("device", get_default_devices())
    def test_split_pad_manual(self, device):
        """handcrafted example to test _split_and_pad_seqeunce"""

        traj = torch.as_tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], device=device)
        splits = torch.as_tensor([3, 2, 1, 4], device=device)
        res = torch.as_tensor(
            [[0, 1, 2, 0], [3, 4, 0, 0], [5, 0, 0, 0], [6, 7, 8, 9]], device=device
        )

        splitted = _split_and_pad_sequence(traj, splits)
        torch.testing.assert_close(res, splitted)

    @pytest.mark.parametrize("B", [(1, ), (3, ), (2, 2, ), (1, 2, 8, )])  # fmt: skip
    @pytest.mark.parametrize("T", [5, 100])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_split_pad_reverse_tensordict(self, B, T, device):
        """calls _split_and_pad_sequence and reverts it on tensordict input"""
        torch.manual_seed(42)

        td = TensorDict(
            {
                "observation": torch.arange(T, dtype=torch.float32, device=device)
                .unsqueeze(-1)
                .expand(*B, T, 3),
                "is_init": torch.zeros(
                    *B, T, 1, dtype=torch.bool, device=device
                ).bernoulli(0.3),
            },
            [*B, T],
            device=device,
        )

        is_init = td.get("is_init").squeeze(-1)
        splits = _get_num_per_traj_init(is_init)
        splitted = _split_and_pad_sequence(
            td.select("observation", strict=False), splits
        )

        reversed = _inv_pad_sequence(splitted, splits)
        reversed = reversed.reshape(td.shape)
        torch.testing.assert_close(td["observation"], reversed["observation"])

    def test_reward2go(self):
        reward = torch.zeros(4, 2)
        reward[3, 0] = 1
        reward[3, 1] = -1
        done = torch.zeros(4, 2, dtype=bool)
        done[3, :] = True
        r = torch.ones(4)
        r[1:] = 0.9
        r = torch.cumprod(r, 0).flip(0)
        r = torch.stack([r, -r], -1)
        torch.testing.assert_close(reward2go(reward, done, 0.9), r)

        reward = torch.zeros(4, 1)
        reward[3, 0] = 1
        done = torch.zeros(4, 1, dtype=bool)
        done[3, :] = True
        r = torch.ones(4)
        r[1:] = 0.9
        reward = reward.expand(2, 4, 1)
        done = done.expand(2, 4, 1)
        r = torch.cumprod(r, 0).flip(0).unsqueeze(-1).expand(2, 4, 1)
        r2go = reward2go(reward, done, 0.9)
        torch.testing.assert_close(r2go, r)

    def test_timedimtranspose_single(self):
        @_transpose_time
        def fun(a, b, time_dim=-2):
            return a + 1

        x = torch.zeros(10)
        y = torch.ones(10)
        with pytest.raises(RuntimeError):
            z = fun(x, y, time_dim=-3)
        with pytest.raises(RuntimeError):
            z = fun(x, y, time_dim=-2)
        z = fun(x, y, time_dim=-1)
        assert z.shape == torch.Size([10])
        assert (z == 1).all()

        @_transpose_time
        def fun(a, b, time_dim=-2):
            return a + 1, b + 1

        with pytest.raises(RuntimeError):
            z1, z2 = fun(x, y, time_dim=-3)
        with pytest.raises(RuntimeError):
            z1, z2 = fun(x, y, time_dim=-2)
        z1, z2 = fun(x, y, time_dim=-1)
        assert z1.shape == torch.Size([10])
        assert (z1 == 1).all()
        assert z2.shape == torch.Size([10])
        assert (z2 == 2).all()


@pytest.mark.parametrize(
    "updater,kwarg",
    [
        (HardUpdate, {"value_network_update_interval": 1000}),
        (SoftUpdate, {"eps": 0.99}),
    ],
)
@set_composite_lp_aggregate(False)
def test_updater_warning(updater, kwarg):
    with warnings.catch_warnings():
        dqn = DQNLoss(torch.nn.Linear(3, 4), delay_value=True, action_space="one_hot")
    with pytest.warns(UserWarning) if rl_warnings() else contextlib.nullcontext():
        dqn.target_value_network_params
    with warnings.catch_warnings():
        updater(dqn, **kwarg)
    with warnings.catch_warnings():
        dqn.target_value_network_params


class TestSingleCall:
    @pytest.fixture(scope="class", autouse=True)
    def _composite_log_prob(self):
        setter = set_composite_lp_aggregate(False)
        setter.set()
        yield
        setter.unset()

    def _mock_value_net(self, has_target, value_key):
        model = nn.Linear(3, 1)
        module = TensorDictModule(model, in_keys=["obs"], out_keys=[value_key])
        params = TensorDict(dict(module.named_parameters()), []).unflatten_keys(".")
        if has_target:
            return (
                module,
                params,
                params.apply(lambda x: x.detach() + torch.randn_like(x)),
            )
        return module, params, params

    def _mock_data(self):
        return TensorDict(
            {
                "obs": torch.randn(10, 3),
                ("next", "obs"): torch.randn(10, 3),
                ("next", "reward"): torch.randn(10, 1),
                ("next", "done"): torch.zeros(10, 1, dtype=torch.bool),
            },
            [10],
            names=["time"],
        )

    @pytest.mark.parametrize("has_target", [True, False])
    @pytest.mark.parametrize("single_call", [True, False])
    @pytest.mark.parametrize("value_key", ["value", ("some", "other", "key")])
    def test_single_call(self, has_target, value_key, single_call, detach_next=True):
        torch.manual_seed(0)
        value_net, params, next_params = self._mock_value_net(has_target, value_key)
        data = self._mock_data()
        if single_call and has_target:
            with pytest.raises(
                ValueError,
                match=r"without recurring to vmap when both params and next params are passed",
            ):
                _call_value_nets(
                    value_net,
                    data,
                    params,
                    next_params,
                    single_call,
                    value_key,
                    detach_next,
                )
            return
        value, value_ = _call_value_nets(
            value_net, data, params, next_params, single_call, value_key, detach_next
        )
        assert (value != value_).all()


@set_composite_lp_aggregate(False)
def test_instantiate_with_different_keys():
    loss_1 = DQNLoss(
        value_network=nn.Linear(3, 3), action_space="one_hot", delay_value=True
    )
    loss_1.set_keys(reward="a")
    assert loss_1.tensor_keys.reward == "a"
    loss_2 = DQNLoss(
        value_network=nn.Linear(3, 3), action_space="one_hot", delay_value=True
    )
    loss_2.set_keys(reward="b")
    assert loss_1.tensor_keys.reward == "a"


class TestBuffer:
    @pytest.fixture(scope="class", autouse=True)
    def _composite_log_prob(self):
        setter = set_composite_lp_aggregate(False)
        setter.set()
        yield
        setter.unset()

    # @pytest.mark.parametrize('dtype', (torch.double, torch.float, torch.half))
    # def test_param_cast(self, dtype):
    #     param = nn.Parameter(torch.zeros(3))
    #     idb = param.data_ptr()
    #     param = param.to(dtype)
    #     assert param.data_ptr() == idb
    #     assert param.dtype == dtype
    #     assert param.data.dtype == dtype
    # @pytest.mark.parametrize('dtype', (torch.double, torch.float, torch.half))
    # def test_buffer_cast(self, dtype):
    #     buffer = Buffer(torch.zeros(3))
    #     idb = buffer.data_ptr()
    #     buffer = buffer.to(dtype)
    #     assert isinstance(buffer, Buffer)
    #     assert buffer.data_ptr() == idb
    #     assert buffer.dtype == dtype
    #     assert buffer.data.dtype == dtype

    @pytest.mark.parametrize("create_target_params", [True, False])
    @pytest.mark.parametrize(
        "dest", [torch.float, torch.double, torch.half, *get_default_devices()]
    )
    def test_module_cast(self, create_target_params, dest):
        # test that when casting a loss module, all the tensors (params and buffers)
        # are properly cast
        class DummyModule(LossModule):
            actor: TensorDictModule
            value: TensorDictModule
            actor_params: TensorDict
            value_params: TensorDict
            target_actor_params: TensorDict
            target_value_params: TensorDict

            def __init__(self):
                common = nn.Linear(3, 4)
                actor = nn.Linear(4, 4)
                value = nn.Linear(4, 1)
                common = TensorDictModule(common, in_keys=["obs"], out_keys=["hidden"])
                actor = TensorDictSequential(
                    common,
                    TensorDictModule(actor, in_keys=["hidden"], out_keys=["action"]),
                )
                value = TensorDictSequential(
                    common,
                    TensorDictModule(value, in_keys=["hidden"], out_keys=["value"]),
                )
                super().__init__()
                self.convert_to_functional(
                    actor,
                    "actor",
                    expand_dim=None,
                    create_target_params=False,
                    compare_against=None,
                )
                self.convert_to_functional(
                    value,
                    "value",
                    expand_dim=2,
                    create_target_params=create_target_params,
                    compare_against=actor.parameters(),
                )

        mod = DummyModule()
        v_p1 = set(mod.value_params.values(True, True)).union(
            set(mod.actor_params.values(True, True))
        )
        v_params1 = set(mod.parameters())
        v_buffers1 = set(mod.buffers())
        mod.to(dest)
        v_p2 = set(mod.value_params.values(True, True)).union(
            set(mod.actor_params.values(True, True))
        )
        v_params2 = set(mod.parameters())
        v_buffers2 = set(mod.buffers())
        assert v_p1 == v_p2
        assert v_params1 == v_params2
        assert v_buffers1 == v_buffers2
        for k, p in mod.named_parameters():
            assert isinstance(p, nn.Parameter), k
        for k, p in mod.named_buffers():
            assert isinstance(p, Buffer), k
        for p in mod.actor_params.values(True, True):
            assert isinstance(p, (nn.Parameter, Buffer))
        for p in mod.value_params.values(True, True):
            assert isinstance(p, (nn.Parameter, Buffer))
        if isinstance(dest, torch.dtype):
            for p in mod.parameters():
                assert p.dtype == dest
            for p in mod.buffers():
                assert p.dtype == dest
            for p in mod.actor_params.values(True, True):
                assert p.dtype == dest
            for p in mod.value_params.values(True, True):
                assert p.dtype == dest
        else:
            for p in mod.parameters():
                assert p.device == dest
            for p in mod.buffers():
                assert p.device == dest
            for p in mod.actor_params.values(True, True):
                assert p.device == dest
            for p in mod.value_params.values(True, True):
                assert p.device == dest


@set_composite_lp_aggregate(False)
def test_loss_exploration():
    class DummyLoss(LossModule):
        def forward(self, td, mode):
            if mode is None:
                mode = self.deterministic_sampling_mode
            assert exploration_type() == mode
            with set_exploration_type(ExplorationType.RANDOM):
                assert exploration_type() == ExplorationType.RANDOM
            assert exploration_type() == mode
            return td

    loss_fn = DummyLoss()
    with set_exploration_type(ExplorationType.RANDOM):
        assert exploration_type() == ExplorationType.RANDOM
        loss_fn(None, None)
        assert exploration_type() == ExplorationType.RANDOM

    with set_exploration_type(ExplorationType.RANDOM):
        assert exploration_type() == ExplorationType.RANDOM
        loss_fn(None, ExplorationType.DETERMINISTIC)
        assert exploration_type() == ExplorationType.RANDOM

    loss_fn.deterministic_sampling_mode = ExplorationType.MODE
    with set_exploration_type(ExplorationType.RANDOM):
        assert exploration_type() == ExplorationType.RANDOM
        loss_fn(None, ExplorationType.MODE)
        assert exploration_type() == ExplorationType.RANDOM

    loss_fn.deterministic_sampling_mode = ExplorationType.MEAN
    with set_exploration_type(ExplorationType.RANDOM):
        assert exploration_type() == ExplorationType.RANDOM
        loss_fn(None, ExplorationType.MEAN)
        assert exploration_type() == ExplorationType.RANDOM


@pytest.mark.parametrize("device", get_default_devices())
class TestMakeValueEstimator:
    """Tests for make_value_estimator accepting ValueEstimatorBase instances and subclasses."""

    def _create_mock_value_net(self, obs_dim=4, device="cpu"):
        """Create a simple value network for testing."""
        return TensorDictModule(
            nn.Linear(obs_dim, 1),
            in_keys=["observation"],
            out_keys=["state_value"],
        ).to(device)

    def _create_mock_actor(self, obs_dim=4, action_dim=2, device="cpu"):
        """Create a simple actor network for testing."""
        return ProbabilisticActor(
            module=TensorDictModule(
                nn.Linear(obs_dim, 2 * action_dim),
                in_keys=["observation"],
                out_keys=["loc", "scale"],
            ),
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            return_log_prob=True,
            spec=Composite(action=Bounded(-1, 1, (action_dim,))),
        ).to(device)

    def _create_mock_qvalue(self, obs_dim=4, action_dim=2, device="cpu"):
        """Create a simple Q-value network for testing."""
        return TensorDictModule(
            nn.Linear(obs_dim + action_dim, 1),
            in_keys=["observation", "action"],
            out_keys=["state_action_value"],
        ).to(device)

    @set_composite_lp_aggregate(False)
    def test_make_value_estimator_with_instance(self, device):
        """Test that make_value_estimator accepts a ValueEstimatorBase instance."""
        value_net = self._create_mock_value_net(device=device)
        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        # Create a value estimator instance
        value_estimator = TD0Estimator(
            gamma=0.99,
            value_network=value_net,
        )

        # Create a loss module that supports value estimation
        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
        )

        # Pass the instance to make_value_estimator
        result = loss_fn.make_value_estimator(value_estimator)

        # Verify the value estimator was set correctly
        assert loss_fn._value_estimator is value_estimator
        assert loss_fn.value_type is TD0Estimator
        # Verify chaining works
        assert result is loss_fn

    @set_composite_lp_aggregate(False)
    def test_make_value_estimator_with_class(self, device):
        """Test that make_value_estimator accepts a ValueEstimatorBase subclass."""
        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        # Create a loss module
        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
        )

        # Pass a class with hyperparameters
        result = loss_fn.make_value_estimator(
            TD0Estimator,
            gamma=0.95,
            value_network=None,  # SAC losses don't need a separate value network
        )

        # Verify the value estimator was instantiated correctly
        assert isinstance(loss_fn._value_estimator, TD0Estimator)
        assert loss_fn.value_type is TD0Estimator
        assert loss_fn._value_estimator.gamma == 0.95
        # Verify chaining works
        assert result is loss_fn

    @set_composite_lp_aggregate(False)
    def test_make_value_estimator_with_class_inherits_device(self, device):
        """Test that make_value_estimator with a class inherits device from loss module."""
        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        # Create a loss module
        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
        )

        # Pass a class without explicit device
        loss_fn.make_value_estimator(
            TD0Estimator,
            gamma=0.99,
            value_network=None,
        )

        # The value estimator should have inherited the device
        assert loss_fn._value_estimator.gamma.device == device

    @set_composite_lp_aggregate(False)
    def test_make_value_estimator_with_gae_class(self, device):
        """Test that make_value_estimator works with GAE class."""
        value_net = self._create_mock_value_net(device=device)
        actor = self._create_mock_actor(device=device)

        # Create a PPO loss which supports GAE
        loss_fn = PPOLoss(
            actor_network=actor,
            critic_network=value_net,
        )

        # Pass GAE class with hyperparameters
        loss_fn.make_value_estimator(
            GAE,
            gamma=0.99,
            lmbda=0.95,
            value_network=value_net,
        )

        # Verify the value estimator was instantiated correctly
        assert isinstance(loss_fn._value_estimator, GAE)
        assert loss_fn.value_type is GAE

    @set_composite_lp_aggregate(False)
    def test_make_value_estimator_with_gae_instance(self, device):
        """Test that make_value_estimator works with GAE instance."""
        value_net = self._create_mock_value_net(device=device)
        actor = self._create_mock_actor(device=device)

        # Create a GAE instance
        gae = GAE(
            gamma=0.99,
            lmbda=0.95,
            value_network=value_net,
        )

        # Create a PPO loss
        loss_fn = PPOLoss(
            actor_network=actor,
            critic_network=value_net,
        )

        # Pass the GAE instance
        loss_fn.make_value_estimator(gae)

        # Verify it was set directly
        assert loss_fn._value_estimator is gae
        assert loss_fn.value_type is GAE


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
