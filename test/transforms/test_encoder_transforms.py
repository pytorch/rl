# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from functools import partial

import pytest

import torch

from _transforms_common import TransformBase
from tensordict import TensorDict, unravel_key
from torch import nn

from torchrl.data import (
    Bounded,
    Composite,
    LazyTensorStorage,
    ReplayBuffer,
    TensorDictReplayBuffer,
    Unbounded,
)
from torchrl.envs import (
    CatTensors,
    Compose,
    gSDENoise,
    ParallelEnv,
    R3MTransform,
    SerialEnv,
    TransformedEnv,
    VC1Transform,
    VIPTransform,
)
from torchrl.envs.libs.gym import _has_gym, GymEnv
from torchrl.envs.transforms.r3m import _R3MNet
from torchrl.envs.transforms.transforms import _has_tv
from torchrl.envs.transforms.vc1 import _has_vc
from torchrl.envs.transforms.vip import _VIPNet, VIPRewardTransform
from torchrl.envs.utils import check_env_specs

from torchrl.testing import (  # noqa
    BREAKOUT_VERSIONED,
    dtype_fixture,
    get_default_devices,
    HALFCHEETAH_VERSIONED,
    PENDULUM_VERSIONED,
    PONG_VERSIONED,
    rand_reset,
    retry,
)
from torchrl.testing.mocking_classes import (
    ContinuousActionVecMockEnv,
    DiscreteActionConvMockEnvNumpy,
)


@pytest.mark.gpu
@pytest.mark.skipif(not _has_tv, reason="torchvision not installed")
@pytest.mark.skipif(not torch.cuda.device_count(), reason="Testing R3M on cuda only")
@pytest.mark.parametrize("device", [torch.device("cuda:0")])
@pytest.mark.parametrize(
    "model",
    [
        "resnet18",
    ],
)  # 1226: "resnet34", "resnet50"])
class TestR3M(TransformBase):
    def test_transform_inverse(self, model, device):
        raise pytest.skip("no inverse for R3MTransform")

    def test_transform_compose(self, model, device):
        if model != "resnet18":
            # we don't test other resnets for the sake of speed and we don't use skip
            # to avoid polluting the log with it
            return
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        r3m = Compose(
            R3MTransform(
                model,
                in_keys=in_keys,
                out_keys=out_keys,
                tensor_pixels_keys=tensor_pixels_key,
            )
        )
        td = TensorDict({"pixels": torch.randint(255, (244, 244, 3))}, [])
        r3m(td)
        assert "vec" in td.keys()
        assert "pixels" not in td.keys()
        assert td["vec"].shape[-1] == 512

    def test_transform_no_env(self, model, device):
        if model != "resnet18":
            # we don't test other resnets for the sake of speed and we don't use skip
            # to avoid polluting the log with it
            return
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        r3m = R3MTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        td = TensorDict({"pixels": torch.randint(255, (244, 244, 3))}, [])
        r3m(td)
        assert "vec" in td.keys()
        assert "pixels" not in td.keys()
        assert td["vec"].shape[-1] == 512

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, model, device, rbclass):
        if model != "resnet18":
            # we don't test other resnets for the sake of speed and we don't use skip
            # to avoid polluting the log with it
            return
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        r3m = R3MTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        rb = rbclass(storage=LazyTensorStorage(20))
        rb.append_transform(r3m)
        td = TensorDict({"pixels": torch.randint(255, (10, 244, 244, 3))}, [10])
        rb.extend(td)
        sample = rb.sample(10)
        assert "vec" in sample.keys()
        assert "pixels" not in sample.keys()
        assert sample["vec"].shape[-1] == 512

    def test_transform_model(self, model, device):
        if model != "resnet18":
            # we don't test other resnets for the sake of speed and we don't use skip
            # to avoid polluting the log with it
            return
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        r3m = R3MTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        td = TensorDict({"pixels": torch.randint(255, (10, 244, 244, 3))}, [10])
        module = nn.Sequential(r3m, nn.Identity())
        sample = module(td)
        assert "vec" in sample.keys()
        assert "pixels" not in sample.keys()
        assert sample["vec"].shape[-1] == 512

    def test_parallel_trans_env_check(self, model, device):
        if model != "resnet18":
            # we don't test other resnets for the sake of speed and we don't use skip
            # to avoid polluting the log with it
            return
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]

        def make_env():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy().to(device),
                R3MTransform(
                    model,
                    in_keys=in_keys,
                    out_keys=out_keys,
                    tensor_pixels_keys=tensor_pixels_key,
                ),
            )

        transformed_env = ParallelEnv(2, make_env)
        try:
            check_env_specs(transformed_env)
        finally:
            transformed_env.close()

    def test_serial_trans_env_check(self, model, device):
        if model != "resnet18":
            # we don't test other resnets for the sake of speed and we don't use skip
            # to avoid polluting the log with it
            return
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]

        def make_env():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy().to(device),
                R3MTransform(
                    model,
                    in_keys=in_keys,
                    out_keys=out_keys,
                    tensor_pixels_keys=tensor_pixels_key,
                ),
            )

        transformed_env = SerialEnv(2, make_env)
        check_env_specs(transformed_env)

    def test_trans_parallel_env_check(self, model, device):
        if model != "resnet18":
            # we don't test other resnets for the sake of speed and we don't use skip
            # to avoid polluting the log with it
            return
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        r3m = R3MTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        transformed_env = TransformedEnv(
            ParallelEnv(2, partial(DiscreteActionConvMockEnvNumpy, device=device)), r3m
        )
        try:
            check_env_specs(transformed_env)
        finally:
            transformed_env.close()

    def test_trans_serial_env_check(self, model, device):
        if model != "resnet18":
            # we don't test other resnets for the sake of speed and we don't use skip
            # to avoid polluting the log with it
            return
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        r3m = R3MTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        transformed_env = TransformedEnv(
            SerialEnv(2, lambda: DiscreteActionConvMockEnvNumpy().to(device)), r3m
        )
        check_env_specs(transformed_env)

    def test_single_trans_env_check(self, model, device):
        if model != "resnet18":
            # we don't test other resnets for the sake of speed and we don't use skip
            # to avoid polluting the log with it
            return
        tensor_pixels_key = None
        in_keys = ["pixels"]
        out_keys = ["vec"]
        r3m = R3MTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        transformed_env = TransformedEnv(
            DiscreteActionConvMockEnvNumpy().to(device), r3m
        )
        check_env_specs(transformed_env)

    @pytest.mark.parametrize("tensor_pixels_key", [None, ["funny_key"]])
    def test_transform_env(self, model, tensor_pixels_key, device):
        in_keys = ["pixels"]
        out_keys = ["vec"]
        r3m = R3MTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        base_env = DiscreteActionConvMockEnvNumpy().to(device)
        transformed_env = TransformedEnv(base_env, r3m)
        td = transformed_env.reset()
        assert td.device == device
        expected_keys = {"vec", "done", "pixels_orig", "terminated"}
        if tensor_pixels_key:
            expected_keys.add(tensor_pixels_key[0])
        assert set(td.keys()) == expected_keys, set(td.keys()) - expected_keys

        td = transformed_env.rand_step(td)
        expected_keys = expected_keys.union(
            {
                ("next", "vec"),
                ("next", "pixels_orig"),
                "action",
                ("next", "reward"),
                ("next", "done"),
                ("next", "terminated"),
                "next",
            }
        )
        if tensor_pixels_key:
            expected_keys.add(("next", tensor_pixels_key[0]))
        assert set(td.keys(True)) == expected_keys, set(td.keys(True)) - expected_keys
        transformed_env.close()

    @pytest.mark.parametrize("stack_images", [True, False])
    @pytest.mark.parametrize(
        "parallel",
        [
            True,
            False,
        ],
    )
    def test_r3m_mult_images(self, model, device, stack_images, parallel):
        in_keys = ["pixels", "pixels2"]
        out_keys = ["vec"] if stack_images else ["vec", "vec2"]
        r3m = R3MTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            stack_images=stack_images,
        )

        def base_env_constructor():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy().to(device),
                CatTensors(["pixels"], "pixels2", del_keys=False),
            )

        assert base_env_constructor().device == device
        if parallel:
            base_env = ParallelEnv(2, base_env_constructor)
        else:
            base_env = base_env_constructor()
        assert base_env.device == device

        transformed_env = TransformedEnv(base_env, r3m)
        assert transformed_env.device == device
        assert r3m.device == device

        td = transformed_env.reset()
        assert td.device == device
        if stack_images:
            expected_keys = {"pixels_orig", "done", "vec", "terminated"}
            # assert td["vec"].shape[0] == 2
            assert td["vec"].ndimension() == 1 + parallel
            assert set(td.keys()) == expected_keys
        else:
            expected_keys = {"pixels_orig", "done", "vec", "vec2", "terminated"}
            assert td["vec"].shape[0 + parallel] != 2
            assert td["vec"].ndimension() == 1 + parallel
            assert td["vec2"].shape[0 + parallel] != 2
            assert td["vec2"].ndimension() == 1 + parallel
            assert set(td.keys()) == expected_keys

        td = transformed_env.rand_step(td)
        expected_keys = expected_keys.union(
            {
                ("next", "vec"),
                ("next", "pixels_orig"),
                "action",
                ("next", "reward"),
                ("next", "done"),
                ("next", "terminated"),
                "next",
            }
        )
        if not stack_images:
            expected_keys.add(("next", "vec2"))
        assert set(td.keys(True)) == expected_keys, set(td.keys()) - expected_keys
        transformed_env.close()

    def test_r3m_parallel(self, model, device):
        in_keys = ["pixels"]
        out_keys = ["vec"]
        tensor_pixels_key = None
        r3m = R3MTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        base_env = ParallelEnv(
            4, partial(DiscreteActionConvMockEnvNumpy, device=device)
        )
        transformed_env = TransformedEnv(base_env, r3m)
        td = transformed_env.reset()
        assert td.device == device
        assert td.batch_size == torch.Size([4])
        expected_keys = {"vec", "done", "pixels_orig", "terminated"}
        if tensor_pixels_key:
            expected_keys.add(tensor_pixels_key)
        assert set(td.keys(True)) == expected_keys

        td = transformed_env.rand_step(td)
        expected_keys = expected_keys.union(
            {
                ("next", "vec"),
                ("next", "pixels_orig"),
                "action",
                ("next", "reward"),
                ("next", "done"),
                ("next", "terminated"),
                "next",
            }
        )
        assert set(td.keys(True)) == expected_keys, set(td.keys()) - expected_keys
        transformed_env.close()
        del transformed_env

    @pytest.mark.parametrize("del_keys", [True, False])
    @pytest.mark.parametrize(
        "in_keys",
        [["pixels"], ["pixels_1", "pixels_2", "pixels_3"]],
    )
    @pytest.mark.parametrize(
        "out_keys",
        [["r3m_vec"], ["r3m_vec_1", "r3m_vec_2", "r3m_vec_3"]],
    )
    def test_r3mnet_transform_observation_spec(
        self, in_keys, out_keys, del_keys, device, model
    ):
        r3m_net = _R3MNet(in_keys, out_keys, model, del_keys)

        observation_spec = Composite(
            {key: Bounded(-1, 1, (3, 16, 16), device) for key in in_keys}
        )
        if del_keys:
            exp_ts = Composite(
                {key: Unbounded(r3m_net.outdim, device) for key in out_keys}
            )

            observation_spec_out = r3m_net.transform_observation_spec(
                observation_spec.clone()
            )

            for key in in_keys:
                assert key not in observation_spec_out
            for key in out_keys:
                assert observation_spec_out[key].shape == exp_ts[key].shape
                assert observation_spec_out[key].device == exp_ts[key].device
                assert observation_spec_out[key].dtype == exp_ts[key].dtype
        else:
            ts_dict = {}
            for key in in_keys:
                ts_dict[key] = observation_spec[key]
            for key in out_keys:
                ts_dict[key] = Unbounded(r3m_net.outdim, device)
            exp_ts = Composite(ts_dict)

            observation_spec_out = r3m_net.transform_observation_spec(
                observation_spec.clone()
            )

            for key in in_keys + out_keys:
                assert observation_spec_out[key].shape == exp_ts[key].shape
                assert observation_spec_out[key].dtype == exp_ts[key].dtype
                assert observation_spec_out[key].device == exp_ts[key].device

    @pytest.mark.parametrize("tensor_pixels_key", [None, ["funny_key"]])
    def test_r3m_spec_against_real(self, model, tensor_pixels_key, device):
        in_keys = ["pixels"]
        out_keys = ["vec"]
        r3m = R3MTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        base_env = DiscreteActionConvMockEnvNumpy().to(device)
        transformed_env = TransformedEnv(base_env, r3m)
        expected_keys = (
            list(transformed_env.state_spec.keys())
            + list(transformed_env.observation_spec.keys())
            + ["action"]
            + [("next", key) for key in transformed_env.observation_spec.keys()]
            + [
                ("next", "reward"),
                ("next", "done"),
                ("next", "terminated"),
                "terminated",
                "done",
                "next",
            ]
        )
        assert set(expected_keys) == set(transformed_env.rollout(3).keys(True))


class TestgSDE(TransformBase):
    @pytest.mark.parametrize("action_dim,state_dim", [(None, None), (7, 7)])
    def test_single_trans_env_check(self, action_dim, state_dim):
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            gSDENoise(state_dim=state_dim, action_dim=action_dim),
        )
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            state_dim = 7
            action_dim = 7
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                gSDENoise(state_dim=state_dim, action_dim=action_dim),
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            state_dim = 7
            action_dim = 7
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                gSDENoise(state_dim=state_dim, action_dim=action_dim),
            )

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize("shape", [(), (2,)])
    def test_trans_serial_env_check(self, shape):
        state_dim = 7
        action_dim = 7
        env = TransformedEnv(
            SerialEnv(2, ContinuousActionVecMockEnv),
            gSDENoise(
                state_dim=state_dim,
                action_dim=action_dim,
                shape=shape,
                expand_specs=True,
            ),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        state_dim = 7
        action_dim = 7
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, ContinuousActionVecMockEnv),
            gSDENoise(state_dim=state_dim, action_dim=action_dim, shape=(2,)),
        )
        try:
            check_env_specs(env)
        finally:
            env.close(raise_if_closed=False)

    def test_transform_no_env(self):
        state_dim = 7
        action_dim = 5
        t = gSDENoise(state_dim=state_dim, action_dim=action_dim, shape=(2,))
        td = TensorDict({"a": torch.zeros(())}, [])
        t(td)
        assert "_eps_gSDE" in td.keys()
        assert (td["_eps_gSDE"] != 0.0).all()
        assert td["_eps_gSDE"].shape == torch.Size(
            [
                2,
                action_dim,
                state_dim,
            ]
        )

    def test_transform_model(self):
        state_dim = 7
        action_dim = 5
        t = gSDENoise(state_dim=state_dim, action_dim=action_dim, shape=(2,))
        model = nn.Sequential(t, nn.Identity())
        td = TensorDict()
        model(td)
        assert "_eps_gSDE" in td.keys()
        assert (td["_eps_gSDE"] != 0.0).all()
        assert td["_eps_gSDE"].shape == torch.Size([2, action_dim, state_dim])

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        state_dim = 7
        action_dim = 5
        batch_size = (2,)
        t = gSDENoise(state_dim=state_dim, action_dim=action_dim, shape=batch_size)
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(t)
        td = TensorDict({"a": torch.zeros(())}, [])
        rb.extend(td.expand(10))
        td = rb.sample(*batch_size)
        assert "_eps_gSDE" in td.keys()
        assert (td["_eps_gSDE"] != 0.0).all()
        assert td["_eps_gSDE"].shape == torch.Size([2, action_dim, state_dim])

    def test_transform_inverse(self):
        raise pytest.skip("No inverse method for TensorDictPrimer")

    def test_transform_compose(self):
        state_dim = 7
        action_dim = 5
        t = Compose(gSDENoise(state_dim=state_dim, action_dim=action_dim, shape=(2,)))
        td = TensorDict({"a": torch.zeros(())}, [])
        t(td)
        assert "_eps_gSDE" in td.keys()
        assert (td["_eps_gSDE"] != 0.0).all()
        assert td["_eps_gSDE"].shape == torch.Size([2, action_dim, state_dim])

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    def test_transform_env(self):
        env = TransformedEnv(
            GymEnv(PENDULUM_VERSIONED()), gSDENoise(state_dim=3, action_dim=1)
        )
        check_env_specs(env)
        assert (env.reset()["_eps_gSDE"] != 0.0).all()


@pytest.mark.gpu
@pytest.mark.skipif(not _has_tv, reason="torchvision not installed")
@pytest.mark.skipif(not torch.cuda.device_count(), reason="Testing VIP on cuda only")
@pytest.mark.parametrize("device", [torch.device("cuda:0")])
@pytest.mark.parametrize("model", ["resnet50"])
class TestVIP(TransformBase):
    def test_transform_inverse(self, model, device):
        raise pytest.skip("no inverse for VIPTransform")

    def test_single_trans_env_check(self, model, device):
        tensor_pixels_key = None
        in_keys = ["pixels"]
        out_keys = ["vec"]
        vip = VIPTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        transformed_env = TransformedEnv(
            DiscreteActionConvMockEnvNumpy().to(device), vip
        )
        check_env_specs(transformed_env)

    def test_trans_serial_env_check(self, model, device):
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        vip = VIPTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        transformed_env = TransformedEnv(
            SerialEnv(2, lambda: DiscreteActionConvMockEnvNumpy().to(device)), vip
        )
        check_env_specs(transformed_env)

    def test_trans_parallel_env_check(self, model, device):
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        vip = VIPTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        transformed_env = TransformedEnv(
            ParallelEnv(2, partial(DiscreteActionConvMockEnvNumpy, device=device)), vip
        )
        try:
            check_env_specs(transformed_env)
        finally:
            transformed_env.close()

    def test_serial_trans_env_check(self, model, device):
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]

        def make_env():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy().to(device),
                VIPTransform(
                    model,
                    in_keys=in_keys,
                    out_keys=out_keys,
                    tensor_pixels_keys=tensor_pixels_key,
                ),
            )

        transformed_env = SerialEnv(2, make_env)
        check_env_specs(transformed_env)

    def test_parallel_trans_env_check(self, model, device):
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]

        def make_env():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy().to(device),
                VIPTransform(
                    model,
                    in_keys=in_keys,
                    out_keys=out_keys,
                    tensor_pixels_keys=tensor_pixels_key,
                ),
            )

        transformed_env = ParallelEnv(2, make_env)
        try:
            check_env_specs(transformed_env)
        finally:
            transformed_env.close()

    def test_transform_model(self, model, device):
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        vip = VIPTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        td = TensorDict({"pixels": torch.randint(255, (10, 244, 244, 3))}, [10])
        module = nn.Sequential(vip, nn.Identity())
        sample = module(td)
        assert "vec" in sample.keys()
        assert "pixels" not in sample.keys()
        assert sample["vec"].shape[-1] == 1024

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, model, device, rbclass):
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        vip = VIPTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        rb = rbclass(storage=LazyTensorStorage(20))
        rb.append_transform(vip)
        td = TensorDict({"pixels": torch.randint(255, (10, 244, 244, 3))}, [10])
        rb.extend(td)
        sample = rb.sample(10)
        assert "vec" in sample.keys()
        assert "pixels" not in sample.keys()
        assert sample["vec"].shape[-1] == 1024

    def test_transform_no_env(self, model, device):
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        vip = VIPTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        td = TensorDict({"pixels": torch.randint(255, (244, 244, 3))}, [])
        vip(td)
        assert "vec" in td.keys()
        assert "pixels" not in td.keys()
        assert td["vec"].shape[-1] == 1024

    def test_transform_compose(self, model, device):
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        vip = Compose(
            VIPTransform(
                model,
                in_keys=in_keys,
                out_keys=out_keys,
                tensor_pixels_keys=tensor_pixels_key,
            )
        )
        td = TensorDict({"pixels": torch.randint(255, (244, 244, 3))}, [])
        vip(td)
        assert "vec" in td.keys()
        assert "pixels" not in td.keys()
        assert td["vec"].shape[-1] == 1024

    @pytest.mark.parametrize("tensor_pixels_key", [None, ["funny_key"]])
    def test_vip_instantiation(self, model, tensor_pixels_key, device):
        in_keys = ["pixels"]
        out_keys = ["vec"]
        vip = VIPTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        base_env = DiscreteActionConvMockEnvNumpy().to(device)
        transformed_env = TransformedEnv(base_env, vip)
        td = transformed_env.reset()
        assert td.device == device
        expected_keys = {"vec", "done", "pixels_orig", "terminated"}
        if tensor_pixels_key:
            expected_keys.add(tensor_pixels_key[0])
        assert set(td.keys()) == expected_keys, set(td.keys()) - expected_keys

        td = transformed_env.rand_step(td)
        expected_keys = expected_keys.union(
            {
                ("next", "vec"),
                ("next", "pixels_orig"),
                "next",
                "action",
                ("next", "reward"),
                ("next", "done"),
                ("next", "terminated"),
            }
        )
        if tensor_pixels_key:
            expected_keys.add(("next", tensor_pixels_key[0]))
        assert set(td.keys(True)) == expected_keys, set(td.keys(True)) - expected_keys
        transformed_env.close()

    @pytest.mark.parametrize("stack_images", [True, False])
    @pytest.mark.parametrize("parallel", [True, False])
    def test_vip_mult_images(self, model, device, stack_images, parallel):
        in_keys = ["pixels", "pixels2"]
        out_keys = ["vec"] if stack_images else ["vec", "vec2"]
        vip = VIPTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            stack_images=stack_images,
        )

        def base_env_constructor():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy().to(device),
                CatTensors(["pixels"], "pixels2", del_keys=False),
            )

        assert base_env_constructor().device == device
        if parallel:
            base_env = ParallelEnv(2, base_env_constructor)
        else:
            base_env = base_env_constructor()
        assert base_env.device == device

        transformed_env = TransformedEnv(base_env, vip)
        assert transformed_env.device == device
        assert vip.device == device

        td = transformed_env.reset()
        assert td.device == device
        if stack_images:
            expected_keys = {"pixels_orig", "done", "vec", "terminated"}
            # assert td["vec"].shape[0] == 2
            assert td["vec"].ndimension() == 1 + parallel
            assert set(td.keys()) == expected_keys
        else:
            expected_keys = {"pixels_orig", "done", "vec", "vec2", "terminated"}
            assert td["vec"].shape[0 + parallel] != 2
            assert td["vec"].ndimension() == 1 + parallel
            assert td["vec2"].shape[0 + parallel] != 2
            assert td["vec2"].ndimension() == 1 + parallel
            assert set(td.keys()) == expected_keys

        td = transformed_env.rand_step(td)
        expected_keys = expected_keys.union(
            {
                ("next", "vec"),
                ("next", "pixels_orig"),
                "next",
                "action",
                ("next", "reward"),
                ("next", "done"),
                ("next", "terminated"),
            }
        )
        if not stack_images:
            expected_keys.add(("next", "vec2"))
        assert set(td.keys(True)) == expected_keys, set(td.keys(True)) - expected_keys
        transformed_env.close()

    def test_transform_env(self, model, device):
        in_keys = ["pixels"]
        out_keys = ["vec"]
        tensor_pixels_key = None
        vip = VIPTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        base_env = ParallelEnv(
            4, partial(DiscreteActionConvMockEnvNumpy, device=device)
        )
        transformed_env = TransformedEnv(base_env, vip)
        td = transformed_env.reset()
        assert td.device == device
        assert td.batch_size == torch.Size([4])
        expected_keys = {"vec", "done", "pixels_orig", "terminated"}
        if tensor_pixels_key:
            expected_keys.add(tensor_pixels_key)
        assert set(td.keys()) == expected_keys

        td = transformed_env.rand_step(td)
        expected_keys = expected_keys.union(
            {
                ("next", "vec"),
                ("next", "pixels_orig"),
                "next",
                "action",
                ("next", "reward"),
                ("next", "done"),
                ("next", "terminated"),
            }
        )
        assert set(td.keys(True)) == expected_keys, set(td.keys(True)) - expected_keys
        transformed_env.close()
        del transformed_env

    def test_vip_parallel_reward(self, model, device, dtype_fixture):  # noqa
        torch.manual_seed(1)
        in_keys = ["pixels"]
        out_keys = ["vec"]
        tensor_pixels_key = None
        vip = VIPRewardTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        base_env = ParallelEnv(
            4, partial(DiscreteActionConvMockEnvNumpy, device=device)
        )
        transformed_env = TransformedEnv(base_env, vip)
        tensordict_reset = TensorDict(
            {"goal_image": torch.randint(0, 255, (4, 7, 7, 3), dtype=torch.uint8)},
            [4],
            device=device,
        )
        with pytest.raises(
            KeyError,
            match=r"VIPRewardTransform.* requires .* key to be present in the input tensordict",
        ):
            _ = transformed_env.reset()
        with pytest.raises(
            KeyError,
            match=r"VIPRewardTransform.* requires .* key to be present in the input tensordict",
        ):
            _ = transformed_env.reset(tensordict_reset.empty())

        td = transformed_env.reset(tensordict_reset)
        assert td.device == device
        assert td.batch_size == torch.Size([4])
        expected_keys = {
            "vec",
            "done",
            "pixels_orig",
            "goal_embedding",
            "goal_image",
            "terminated",
        }
        if tensor_pixels_key:
            expected_keys.add(tensor_pixels_key)
        assert set(td.keys()) == expected_keys

        td = transformed_env.rand_step(td)
        expected_keys = expected_keys.union(
            {
                ("next", "vec"),
                ("next", "pixels_orig"),
                "next",
                "action",
                ("next", "reward"),
                ("next", "done"),
                ("next", "terminated"),
            }
        )
        assert set(td.keys(True)) == expected_keys, td

        torch.manual_seed(1)
        tensordict_reset = TensorDict(
            {"goal_image": torch.randint(0, 255, (4, 7, 7, 3), dtype=torch.uint8)},
            [4],
            device=device,
        )
        td = transformed_env.rollout(
            5, auto_reset=False, tensordict=transformed_env.reset(tensordict_reset)
        )
        assert set(td.keys(True)) == expected_keys, td
        # test that we do compute the reward we want
        cur_embedding = td["next", "vec"]
        goal_embedding = td["goal_embedding"]
        last_embedding = td["vec"]

        # test that there is only one goal embedding
        goal = td["goal_embedding"]
        goal_expand = td["goal_embedding"][:, :1].expand_as(td["goal_embedding"])
        torch.testing.assert_close(goal, goal_expand)

        torch.testing.assert_close(cur_embedding[:, :-1], last_embedding[:, 1:])
        with pytest.raises(AssertionError):
            torch.testing.assert_close(cur_embedding[:, 1:], last_embedding[:, :-1])

        explicit_reward = -torch.linalg.norm(cur_embedding - goal_embedding, dim=-1) - (
            -torch.linalg.norm(last_embedding - goal_embedding, dim=-1)
        )
        torch.testing.assert_close(explicit_reward, td["next", "reward"].squeeze())

        transformed_env.close()
        del transformed_env

    @pytest.mark.parametrize("del_keys", [True, False])
    @pytest.mark.parametrize(
        "in_keys",
        [["pixels"], ["pixels_1", "pixels_2", "pixels_3"]],
    )
    @pytest.mark.parametrize(
        "out_keys",
        [["vip_vec"], ["vip_vec_1", "vip_vec_2", "vip_vec_3"]],
    )
    def test_vipnet_transform_observation_spec(
        self, in_keys, out_keys, del_keys, device, model
    ):
        vip_net = _VIPNet(in_keys, out_keys, model, del_keys)

        observation_spec = Composite(
            {key: Bounded(-1, 1, (3, 16, 16), device) for key in in_keys}
        )
        if del_keys:
            exp_ts = Composite({key: Unbounded(1024, device) for key in out_keys})

            observation_spec_out = vip_net.transform_observation_spec(
                observation_spec.clone()
            )

            for key in in_keys:
                assert key not in observation_spec_out
            for key in out_keys:
                assert observation_spec_out[key].shape == exp_ts[key].shape
                assert observation_spec_out[key].device == exp_ts[key].device
                assert observation_spec_out[key].dtype == exp_ts[key].dtype
        else:
            ts_dict = {}
            for key in in_keys:
                ts_dict[key] = observation_spec[key]
            for key in out_keys:
                ts_dict[key] = Unbounded(1024, device)
            exp_ts = Composite(ts_dict)

            observation_spec_out = vip_net.transform_observation_spec(
                observation_spec.clone()
            )

            for key in in_keys + out_keys:
                assert observation_spec_out[key].shape == exp_ts[key].shape
                assert observation_spec_out[key].dtype == exp_ts[key].dtype
                assert observation_spec_out[key].device == exp_ts[key].device

    @pytest.mark.parametrize("tensor_pixels_key", [None, ["funny_key"]])
    def test_vip_spec_against_real(self, model, tensor_pixels_key, device):
        in_keys = ["pixels"]
        out_keys = ["vec"]
        vip = VIPTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        base_env = DiscreteActionConvMockEnvNumpy().to(device)
        transformed_env = TransformedEnv(base_env, vip)
        expected_keys = (
            list(transformed_env.state_spec.keys())
            + ["action"]
            + list(transformed_env.observation_spec.keys())
            + [("next", key) for key in transformed_env.observation_spec.keys()]
            + [
                ("next", "reward"),
                ("next", "done"),
                "done",
                ("next", "terminated"),
                "terminated",
                "next",
            ]
        )
        assert set(expected_keys) == set(transformed_env.rollout(3).keys(True))


@pytest.mark.gpu
@pytest.mark.skipif(not _has_vc, reason="vc_models not installed")
@pytest.mark.skipif(not torch.cuda.device_count(), reason="VC1 should run on cuda")
@pytest.mark.parametrize("device", [torch.device("cuda:0")])
class TestVC1(TransformBase):
    def test_transform_inverse(self, device):
        raise pytest.skip("no inverse for VC1Transform")

    def test_single_trans_env_check(self, device):
        del_keys = False
        in_keys = ["pixels"]
        out_keys = ["vec"]
        vc1 = VC1Transform(
            in_keys=in_keys,
            out_keys=out_keys,
            del_keys=del_keys,
            model_name="default",
        )
        transformed_env = TransformedEnv(
            DiscreteActionConvMockEnvNumpy().to(device), vc1
        )
        check_env_specs(transformed_env)

    def test_trans_serial_env_check(self, device):
        in_keys = ["pixels"]
        del_keys = False
        out_keys = ["vec"]
        vc1 = VC1Transform(
            in_keys=in_keys,
            out_keys=out_keys,
            del_keys=del_keys,
            model_name="default",
        )
        transformed_env = TransformedEnv(
            SerialEnv(2, lambda: DiscreteActionConvMockEnvNumpy().to(device)), vc1
        )
        check_env_specs(transformed_env)

    def test_trans_parallel_env_check(self, device):
        in_keys = ["pixels"]
        del_keys = False
        out_keys = ["vec"]
        vc1 = VC1Transform(
            in_keys=in_keys,
            out_keys=out_keys,
            del_keys=del_keys,
            model_name="default",
        )
        transformed_env = TransformedEnv(
            ParallelEnv(2, partial(DiscreteActionConvMockEnvNumpy, device=device)), vc1
        )
        try:
            check_env_specs(transformed_env)
        finally:
            transformed_env.close()

    def test_serial_trans_env_check(self, device):
        in_keys = ["pixels"]
        del_keys = False
        out_keys = ["vec"]

        def make_env():
            t = VC1Transform(
                in_keys=in_keys,
                out_keys=out_keys,
                del_keys=del_keys,
                model_name="default",
            )

            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy().to(device),
                t,
            )

        transformed_env = SerialEnv(2, make_env)
        check_env_specs(transformed_env)

    def test_parallel_trans_env_check(self, device):
        # let's spare this one
        return

    def test_transform_model(self, device):
        in_keys = ["pixels"]
        del_keys = False
        out_keys = ["vec"]
        vc1 = VC1Transform(
            in_keys=in_keys,
            out_keys=out_keys,
            del_keys=del_keys,
            model_name="default",
        )
        td = TensorDict({"pixels": torch.randint(255, (10, 244, 244, 3))}, [10])
        module = nn.Sequential(vc1, nn.Identity())
        sample = module(td)
        assert "vec" in sample.keys()
        if del_keys:
            assert "pixels" not in sample.keys()
        assert sample["vec"].shape[-1] == 16

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, device, rbclass):
        in_keys = ["pixels"]
        del_keys = False
        out_keys = ["vec"]
        vc1 = VC1Transform(
            in_keys=in_keys,
            out_keys=out_keys,
            del_keys=del_keys,
            model_name="default",
        )
        rb = rbclass(storage=LazyTensorStorage(20))
        rb.append_transform(vc1)
        td = TensorDict({"pixels": torch.randint(255, (10, 244, 244, 3))}, [10])
        rb.extend(td)
        sample = rb.sample(10)
        assert "vec" in sample.keys()
        if del_keys:
            assert "pixels" not in sample.keys()
        assert sample["vec"].shape[-1] == 16

    def test_transform_no_env(self, device):
        in_keys = ["pixels"]
        del_keys = False
        out_keys = ["vec"]
        vc1 = VC1Transform(
            in_keys=in_keys,
            out_keys=out_keys,
            del_keys=del_keys,
            model_name="default",
        )
        td = TensorDict({"pixels": torch.randint(255, (244, 244, 3))}, [])
        vc1(td)
        assert "vec" in td.keys()
        if del_keys:
            assert "pixels" not in td.keys()
        assert td["vec"].shape[-1] == 16

    def test_transform_compose(self, device):
        in_keys = ["pixels"]
        del_keys = False
        out_keys = ["vec"]
        vip = Compose(
            VC1Transform(
                in_keys=in_keys,
                out_keys=out_keys,
                del_keys=del_keys,
                model_name="default",
            )
        )
        td = TensorDict({"pixels": torch.randint(255, (244, 244, 3))}, [])
        vip(td)
        assert "vec" in td.keys()
        if del_keys:
            assert "pixels" not in td.keys()
        assert td["vec"].shape[-1] == 16

    @pytest.mark.parametrize("del_keys", [False, True])
    def test_vc1_instantiation(self, del_keys, device):
        in_keys = ["pixels"]
        out_keys = [("nested", "vec")]
        vc1 = VC1Transform(
            in_keys=in_keys,
            out_keys=out_keys,
            del_keys=del_keys,
            model_name="default",
        )
        base_env = DiscreteActionConvMockEnvNumpy().to(device)
        transformed_env = TransformedEnv(base_env, vc1)
        td = transformed_env.reset()
        assert td.device == device
        expected_keys = {"nested", "done", "pixels_orig", "terminated"}
        if not del_keys:
            expected_keys.add("pixels")
        assert set(td.keys()) == expected_keys, set(td.keys()) - expected_keys

        td = transformed_env.rand_step(td)
        expected_keys = expected_keys.union(
            {
                ("next", "nested"),
                ("next", "nested", "vec"),
                ("next", "pixels_orig"),
                "next",
                "action",
                ("nested", "vec"),
                ("next", "reward"),
                ("next", "done"),
                ("next", "terminated"),
            }
        )
        if not del_keys:
            expected_keys.add(("next", "pixels"))
        assert set(td.keys(True)) == expected_keys, set(td.keys(True)) - expected_keys
        transformed_env.close()

    @pytest.mark.parametrize("del_keys", [True, False])
    def test_transform_env(self, device, del_keys):
        in_keys = ["pixels"]
        out_keys = [("nested", "vec")]

        vc1 = VC1Transform(
            in_keys=in_keys,
            out_keys=out_keys,
            del_keys=del_keys,
            model_name="default",
        )
        base_env = ParallelEnv(
            4, partial(DiscreteActionConvMockEnvNumpy, device=device)
        )
        transformed_env = TransformedEnv(base_env, vc1)
        td = transformed_env.reset()
        assert td.device == device
        assert td.batch_size == torch.Size([4])
        expected_keys = {"nested", "done", "pixels_orig", "terminated"}
        if not del_keys:
            expected_keys.add("pixels")
        assert set(td.keys()) == expected_keys

        td = transformed_env.rand_step(td)
        expected_keys = expected_keys.union(
            {
                ("next", "nested"),
                ("next", "nested", "vec"),
                ("next", "pixels_orig"),
                "next",
                "action",
                ("nested", "vec"),
                ("next", "reward"),
                ("next", "done"),
                ("next", "terminated"),
            }
        )
        if not del_keys:
            expected_keys.add(("next", "pixels"))
        assert set(td.keys(True)) == expected_keys, set(td.keys(True)) - expected_keys
        transformed_env.close()
        del transformed_env

    @pytest.mark.parametrize("del_keys", [True, False])
    def test_vc1_spec_against_real(self, del_keys, device):
        in_keys = ["pixels"]
        out_keys = [("nested", "vec")]
        vc1 = VC1Transform(
            in_keys=in_keys,
            out_keys=out_keys,
            del_keys=del_keys,
            model_name="default",
        )
        base_env = DiscreteActionConvMockEnvNumpy().to(device)
        transformed_env = TransformedEnv(base_env, vc1)
        expected_keys = (
            list(transformed_env.state_spec.keys())
            + ["action"]
            + list(transformed_env.observation_spec.keys(True))
            + [
                unravel_key(("next", key))
                for key in transformed_env.observation_spec.keys(True)
            ]
            + [
                ("next", "reward"),
                ("next", "done"),
                "done",
                ("next", "terminated"),
                "terminated",
                "next",
            ]
        )
        assert set(expected_keys) == set(transformed_env.rollout(3).keys(True))
