# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch
from _objectives_common import LossModuleTestBase

from tensordict import TensorDict
from tensordict.nn import (
    InteractionType,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
)
from torch import nn

from torchrl.data import Unbounded
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.transforms import TensorDictPrimer, TransformedEnv
from torchrl.modules import SafeSequential, WorldModel, WorldModelWrapper
from torchrl.modules.distributions.continuous import TanhNormal
from torchrl.modules.models.model_based import (
    DreamerActor,
    ObsDecoder,
    ObsEncoder,
    RSSMPosterior,
    RSSMPrior,
    RSSMRollout,
)
from torchrl.modules.models.models import MLP
from torchrl.objectives import DreamerActorLoss, DreamerModelLoss, DreamerValueLoss
from torchrl.objectives.utils import ValueEstimators
from torchrl.objectives.world_model_loss import WorldModelLoss

from torchrl.testing import (  # noqa
    call_value_nets as _call_value_nets,
    dtype_fixture,
    get_available_devices,
    get_default_devices,
    PENDULUM_VERSIONED,
)
from torchrl.testing.mocking_classes import ContinuousActionConvMockEnv


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
        if td_est in (
            ValueEstimators.GAE,
            ValueEstimators.MAGAE,
            ValueEstimators.VTrace,
        ):
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


# ---------------------------------------------------------------------------
# WorldModelLoss tests
# (moved here from test/test_world_model.py per review consolidation request)
# ---------------------------------------------------------------------------

_WML_OBS_DIM = 8
_WML_LATENT_DIM = 4
_WML_ACTION_DIM = 2
_WML_BATCH = 3


class _WMLCatLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat(tensors, dim=-1))


def _wml_make_world_model(
    with_done_head: bool = False, with_decoder: bool = False
) -> WorldModel:
    from tensordict.nn import TensorDictModule

    encoder = TensorDictModule(
        torch.nn.Linear(_WML_OBS_DIM, _WML_LATENT_DIM),
        in_keys=["observation"],
        out_keys=["latent"],
    )
    dynamics = TensorDictModule(
        _WMLCatLinear(_WML_LATENT_DIM + _WML_ACTION_DIM, _WML_LATENT_DIM),
        in_keys=["latent", "action"],
        out_keys=[("next", "latent")],
    )
    reward_head = TensorDictModule(
        torch.nn.Linear(_WML_LATENT_DIM, 1),
        in_keys=[("next", "latent")],
        out_keys=[("next", "reward")],
    )
    done_head = None
    if with_done_head:
        done_head = TensorDictModule(
            torch.nn.Linear(_WML_LATENT_DIM, 1),
            in_keys=[("next", "latent")],
            out_keys=[("next", "done")],
        )
    decoder = None
    if with_decoder:
        decoder = TensorDictModule(
            torch.nn.Linear(_WML_LATENT_DIM, _WML_OBS_DIM),
            in_keys=["latent"],
            out_keys=["reconstructed_observation"],
        )
    return WorldModel(
        encoder, dynamics, reward_head, done_head=done_head, decoder=decoder
    )


def _wml_make_real_batch(with_done: bool = False) -> TensorDict:
    data = {
        "observation": torch.randn(_WML_BATCH, _WML_OBS_DIM),
        "action": torch.randn(_WML_BATCH, _WML_ACTION_DIM),
        "next": {
            "reward": torch.randn(_WML_BATCH, 1),
            "latent": torch.randn(_WML_BATCH, _WML_LATENT_DIM),
        },
    }
    if with_done:
        data["next"]["done"] = torch.zeros(_WML_BATCH, 1, dtype=torch.bool)
        data["next"]["terminated"] = torch.zeros(_WML_BATCH, 1, dtype=torch.bool)
    return TensorDict(data, batch_size=[_WML_BATCH])


class TestWorldModelLoss:
    def test_reward_loss_only(self):
        wm = _wml_make_world_model()
        loss = WorldModelLoss(wm, losses=["reward"])
        batch = _wml_make_real_batch()
        td_out = loss(batch)
        assert "loss_reward" in td_out.keys()
        assert td_out["loss_reward"].shape == torch.Size([])

    def test_reconstruction_loss(self):
        wm = _wml_make_world_model(with_decoder=True)
        loss = WorldModelLoss(wm, losses=["reward", "reconstruction"])
        batch = _wml_make_real_batch()
        td_out = loss(batch)
        assert "loss_reward" in td_out.keys()
        assert "loss_reconstruction" in td_out.keys()

    def test_latent_loss(self):
        from tensordict.nn import TensorDictModule

        encoder = TensorDictModule(
            torch.nn.Linear(_WML_OBS_DIM, _WML_LATENT_DIM),
            in_keys=["observation"],
            out_keys=["latent"],
        )
        dynamics = TensorDictModule(
            _WMLCatLinear(_WML_LATENT_DIM + _WML_ACTION_DIM, _WML_LATENT_DIM),
            in_keys=["latent", "action"],
            out_keys=["predicted_latent"],
        )
        reward_head = TensorDictModule(
            torch.nn.Linear(_WML_LATENT_DIM, 1),
            in_keys=["predicted_latent"],
            out_keys=[("next", "reward")],
        )
        wm = WorldModel(encoder, dynamics, reward_head)
        loss = WorldModelLoss(wm, losses=["reward", "latent"])
        batch = TensorDict(
            {
                "observation": torch.randn(_WML_BATCH, _WML_OBS_DIM),
                "action": torch.randn(_WML_BATCH, _WML_ACTION_DIM),
                "predicted_latent": torch.randn(_WML_BATCH, _WML_LATENT_DIM),
                "target_latent": torch.randn(_WML_BATCH, _WML_LATENT_DIM),
                "next": {"reward": torch.randn(_WML_BATCH, 1)},
            },
            batch_size=[_WML_BATCH],
        )
        td_out = loss(batch)
        assert "loss_latent" in td_out.keys()

    def test_unknown_loss_raises(self):
        wm = _wml_make_world_model()
        with pytest.raises(ValueError, match="Unknown loss type"):
            WorldModelLoss(wm, losses=["bad_loss"])

    def test_set_keys(self):
        wm = _wml_make_world_model()
        loss = WorldModelLoss(wm, losses=["reward"])
        loss.set_keys(reward="my_reward", true_reward="my_true_reward")
        assert loss.tensor_keys.reward == "my_reward"
        assert loss.tensor_keys.true_reward == "my_true_reward"

    def test_weights_applied(self):
        wm = _wml_make_world_model()
        loss_1x = WorldModelLoss(wm, losses=["reward"], reward_weight=1.0)
        loss_2x = WorldModelLoss(wm, losses=["reward"], reward_weight=2.0)
        batch = _wml_make_real_batch()
        out_1x = loss_1x(batch)
        out_2x = loss_2x(batch)
        assert torch.allclose(out_2x["loss_reward"], 2.0 * out_1x["loss_reward"])

    def test_done_loss(self):
        wm = _wml_make_world_model(with_done_head=True)
        loss = WorldModelLoss(wm, losses=["reward", "done"])
        batch = _wml_make_real_batch(with_done=True)
        td_out = loss(batch)
        assert "loss_done" in td_out.keys()

    def test_loss_is_differentiable(self):
        wm = _wml_make_world_model()
        loss = WorldModelLoss(wm, losses=["reward"])
        batch = _wml_make_real_batch()
        td_out = loss(batch)
        td_out["loss_reward"].backward()
        for p in wm.parameters():
            assert p.grad is not None
