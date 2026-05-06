# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest

import tensordict.tensordict
import torch

from _transforms_common import _has_ale, TransformBase
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.utils import assert_allclose_td
from torch import nn, Tensor
from torchrl._utils import prod

from torchrl.data import (
    Bounded,
    Composite,
    LazyTensorStorage,
    ReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.envs import (
    CatFrames,
    CenterCrop,
    Compose,
    Crop,
    FlattenObservation,
    GrayScale,
    ObservationNorm,
    ParallelEnv,
    PermuteTransform,
    Resize,
    SerialEnv,
    TimeMaxPool,
    ToTensorImage,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.libs.gym import _has_gym, GymEnv
from torchrl.envs.transforms.transforms import _has_tv
from torchrl.envs.utils import check_env_specs, step_mdp

from torchrl.testing import (  # noqa
    BREAKOUT_VERSIONED,
    CARTPOLE_VERSIONED,
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
    CountingEnv,
    CountingEnvCountPolicy,
    DiscreteActionConvMockEnv,
    DiscreteActionConvMockEnvNumpy,
    NestedCountingEnv,
)


class TestCatFrames(TransformBase):
    @pytest.mark.parametrize("out_keys", [None, ["obs2"]])
    def test_single_trans_env_check(self, out_keys):
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            CatFrames(dim=-1, N=3, in_keys=["observation"], out_keys=out_keys),
        )
        check_env_specs(env)

    @pytest.mark.parametrize("cat_dim", [-1, -2, -3])
    @pytest.mark.parametrize("cat_N", [3, 10])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_with_permute_no_env(self, cat_dim, cat_N, device):
        torch.manual_seed(cat_dim * cat_N)
        pixels = torch.randn(8, 5, 3, 10, 4, device=device)

        a = TensorDict(
            {
                "pixels": pixels,
            },
            [
                pixels.shape[0],
            ],
            device=device,
        )

        t0 = Compose(
            CatFrames(N=cat_N, dim=cat_dim),
        )

        def get_rand_perm(ndim):
            cat_dim_perm = cat_dim
            # Ensure that the permutation moves the cat_dim
            while cat_dim_perm == cat_dim:
                perm_pos = torch.randperm(ndim)
                perm = perm_pos - ndim
                cat_dim_perm = (perm == cat_dim).nonzero().item() - ndim
                perm_inv = perm_pos.argsort() - ndim
            return perm.tolist(), perm_inv.tolist(), cat_dim_perm

        perm, perm_inv, cat_dim_perm = get_rand_perm(pixels.dim() - 1)

        t1 = Compose(
            PermuteTransform(perm, in_keys=["pixels"]),
            CatFrames(N=cat_N, dim=cat_dim_perm),
            PermuteTransform(perm_inv, in_keys=["pixels"]),
        )

        b = t0._call(a.clone())
        c = t1._call(a.clone())
        assert (b == c).all()

    @pytest.mark.skipif(not _has_gym, reason="Test executed on gym")
    @pytest.mark.parametrize("cat_dim", [-1, -2])
    def test_with_permute_env(self, cat_dim):
        env0 = TransformedEnv(
            GymEnv(PENDULUM_VERSIONED()),
            Compose(
                UnsqueezeTransform(-1, in_keys=["observation"]),
                CatFrames(N=4, dim=cat_dim, in_keys=["observation"]),
            ),
        )

        env1 = TransformedEnv(
            GymEnv(PENDULUM_VERSIONED()),
            Compose(
                UnsqueezeTransform(-1, in_keys=["observation"]),
                PermuteTransform((-1, -2), in_keys=["observation"]),
                CatFrames(N=4, dim=-3 - cat_dim, in_keys=["observation"]),
                PermuteTransform((-1, -2), in_keys=["observation"]),
            ),
        )

        torch.manual_seed(0)
        env0.set_seed(0)
        td0 = env0.reset()

        torch.manual_seed(0)
        env1.set_seed(0)
        td1 = env1.reset()

        assert (td0 == td1).all()

        td0 = env0.step(td0.update(env0.full_action_spec.rand()))
        td1 = env0.step(td0.update(env1.full_action_spec.rand()))

        assert (td0 == td1).all()

    def test_serial_trans_env_check(self):
        env = SerialEnv(
            2,
            lambda: TransformedEnv(
                ContinuousActionVecMockEnv(),
                CatFrames(dim=-1, N=3, in_keys=["observation"]),
            ),
        )
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        env = maybe_fork_ParallelEnv(
            2,
            lambda: TransformedEnv(
                ContinuousActionVecMockEnv(),
                CatFrames(dim=-1, N=3, in_keys=["observation"]),
            ),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, lambda: ContinuousActionVecMockEnv()),
            CatFrames(dim=-1, N=3, in_keys=["observation"]),
        )
        check_env_specs(env)
        env2 = SerialEnv(
            2,
            lambda: TransformedEnv(
                ContinuousActionVecMockEnv(),
                CatFrames(dim=-1, N=3, in_keys=["observation"]),
            ),
        )

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, ContinuousActionVecMockEnv),
            CatFrames(dim=-1, N=3, in_keys=["observation"]),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.skipif(not _has_gym, reason="Test executed on gym")
    @pytest.mark.parametrize("batched_class", [ParallelEnv, SerialEnv])
    @pytest.mark.parametrize("break_when_any_done", [True, False])
    def test_catframes_batching(
        self, batched_class, break_when_any_done, maybe_fork_ParallelEnv
    ):
        if batched_class is ParallelEnv:
            batched_class = maybe_fork_ParallelEnv

        env = TransformedEnv(
            batched_class(2, lambda: GymEnv(CARTPOLE_VERSIONED())),
            CatFrames(
                dim=-1, N=3, in_keys=["observation"], out_keys=["observation_cat"]
            ),
        )
        torch.manual_seed(0)
        env.set_seed(0)
        r0 = env.rollout(100, break_when_any_done=break_when_any_done)

        env = batched_class(
            2,
            lambda: TransformedEnv(
                GymEnv(CARTPOLE_VERSIONED()),
                CatFrames(
                    dim=-1, N=3, in_keys=["observation"], out_keys=["observation_cat"]
                ),
            ),
        )
        torch.manual_seed(0)
        env.set_seed(0)
        r1 = env.rollout(100, break_when_any_done=break_when_any_done)
        tensordict.tensordict.assert_allclose_td(r0, r1)

    def test_nested(self, nested_dim=3, batch_size=(32, 1), rollout_length=6, cat_N=5):
        env = NestedCountingEnv(
            max_steps=20, nested_dim=nested_dim, batch_size=batch_size
        )
        policy = CountingEnvCountPolicy(
            action_spec=env.full_action_spec[env.action_key], action_key=env.action_key
        )
        td = env.rollout(rollout_length, policy=policy)
        assert td[("data", "states")].shape == (
            *batch_size,
            rollout_length,
            nested_dim,
            1,
        )
        transformed_env = TransformedEnv(
            env, CatFrames(dim=-1, N=cat_N, in_keys=[("data", "states")])
        )
        td = transformed_env.rollout(rollout_length, policy=policy)
        assert td[("data", "states")].shape == (
            *batch_size,
            rollout_length,
            nested_dim,
            cat_N,
        )
        assert (
            (td[("data", "states")][0, 0, -1, 0]).eq(torch.arange(1, 1 + cat_N)).all()
        )
        assert (
            (td[("next", "data", "states")][0, 0, -1, 0])
            .eq(torch.arange(2, 2 + cat_N))
            .all()
        )

    @pytest.mark.skipif(not _has_gym, reason="Gym not available")
    def test_transform_env(self):
        env = TransformedEnv(
            GymEnv(PENDULUM_VERSIONED(), frame_skip=4),
            CatFrames(dim=-1, N=3, in_keys=["observation"]),
        )
        td = env.reset()
        assert td["observation"].shape[-1] == 9
        assert (td["observation"][..., :3] == td["observation"][..., 3:6]).all()
        assert (td["observation"][..., 3:6] == td["observation"][..., 6:9]).all()
        old = td["observation"][..., 3:6].clone()
        td = env.rand_step(td)
        assert (td["next", "observation"][..., :3] == old).all()
        assert (
            td["next", "observation"][..., :3] == td["next", "observation"][..., 3:6]
        ).all()
        assert (
            td["next", "observation"][..., 3:6] != td["next", "observation"][..., 6:9]
        ).any()

    @pytest.mark.skipif(not _has_gym, reason="Gym not available")
    def test_transform_env_clone(self):
        env = TransformedEnv(
            GymEnv(PENDULUM_VERSIONED(), frame_skip=4),
            CatFrames(dim=-1, N=3, in_keys=["observation"]),
        )
        td = env.reset()
        td = env.rand_step(td)
        cloned = env.transform.clone()
        value_at_clone = td["next", "observation"].clone()
        for _ in range(10):
            td = env.rand_step(td)
            td = step_mdp(td)
        assert (td["observation"] != value_at_clone).any()
        assert (td["observation"] == env.transform._cat_buffers_observation).all()
        assert (
            cloned._cat_buffers_observation == env.transform._cat_buffers_observation
        ).all()
        assert cloned is not env.transform

    @pytest.mark.parametrize("dim", [-1])
    @pytest.mark.parametrize("N", [3, 4])
    @pytest.mark.parametrize("padding", ["constant", "same"])
    def test_transform_model(self, dim, N, padding):
        # test equivalence between transforms within an env and within a rb
        key1 = "observation"
        keys = [key1]
        out_keys = ["out_" + key1]
        cat_frames = CatFrames(
            N=N, in_keys=keys, out_keys=out_keys, dim=dim, padding=padding
        )
        cat_frames2 = CatFrames(
            N=N,
            in_keys=keys + [("next", keys[0])],
            out_keys=out_keys + [("next", out_keys[0])],
            dim=dim,
            padding=padding,
        )
        envbase = ContinuousActionVecMockEnv()
        env = TransformedEnv(envbase, cat_frames)

        torch.manual_seed(10)
        env.set_seed(10)
        td = env.rollout(10)

        torch.manual_seed(10)
        envbase.set_seed(10)
        tdbase = envbase.rollout(10)

        tdbase0 = tdbase.clone()

        model = nn.Sequential(cat_frames2, nn.Identity())
        model(tdbase)
        assert assert_allclose_td(td, tdbase)

        with pytest.warns(UserWarning):
            tdbase0.names = None
            model(tdbase0)
        tdbase0.batch_size = []
        with pytest.raises(
            ValueError, match="CatFrames cannot process unbatched tensordict"
        ):
            model(tdbase0)
        tdbase0.batch_size = [10]
        tdbase0 = tdbase0.expand(5, 10)
        tdbase0_copy = tdbase0.transpose(0, 1)
        tdbase0.refine_names("time", None)
        tdbase0_copy.names = [None, "time"]
        v1 = model(tdbase0)
        v2 = model(tdbase0_copy)
        # check that swapping dims and names leads to same result
        assert_allclose_td(v1, v2.transpose(0, 1))

    @pytest.mark.parametrize("dim", [-1])
    @pytest.mark.parametrize("N", [3, 4])
    @pytest.mark.parametrize("padding", ["same", "constant"])
    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, dim, N, padding, rbclass):
        # test equivalence between transforms within an env and within a rb
        key1 = "observation"
        keys = [key1]
        out_keys = ["out_" + key1]
        cat_frames = CatFrames(
            N=N, in_keys=keys, out_keys=out_keys, dim=dim, padding=padding
        )
        cat_frames2 = CatFrames(
            N=N,
            in_keys=keys + [("next", keys[0])],
            out_keys=out_keys + [("next", out_keys[0])],
            dim=dim,
            padding=padding,
        )

        env = TransformedEnv(ContinuousActionVecMockEnv(), cat_frames)
        td = env.rollout(10)

        rb = rbclass(storage=LazyTensorStorage(20))
        rb.append_transform(cat_frames2)
        rb.add(td.exclude(*out_keys, ("next", out_keys[0])))
        tdsample = rb.sample(1).squeeze(0).exclude("index")
        for key in td.keys(True, True):
            assert (tdsample[key] == td[key]).all(), key
        assert (tdsample["out_" + key1] == td["out_" + key1]).all()
        assert (tdsample["next", "out_" + key1] == td["next", "out_" + key1]).all()

    def test_transform_rb_maker(self):
        env = CountingEnv(max_steps=10)
        catframes = CatFrames(
            in_keys=["observation"], out_keys=["observation_stack"], dim=-1, N=4
        )
        env = env.append_transform(catframes)
        policy = lambda td: td.update(env.full_action_spec.zeros() + 1)
        rollout = env.rollout(150, policy, break_when_any_done=False)
        transform, sampler = catframes.make_rb_transform_and_sampler(batch_size=32)
        rb = ReplayBuffer(
            sampler=sampler, storage=LazyTensorStorage(150), transform=transform
        )
        rb.extend(rollout)
        sample = rb.sample(32)
        assert "observation_stack" not in rb._storage._storage
        assert sample.shape == (32,)
        assert sample["observation_stack"].shape == (32, 4)
        assert sample["next", "observation_stack"].shape == (32, 4)
        assert (
            sample["observation_stack"]
            == sample["observation_stack"][:, :1] + torch.arange(4)
        ).all()

    @pytest.mark.parametrize("dim", [-1])
    @pytest.mark.parametrize("N", [3, 4])
    @pytest.mark.parametrize("padding", ["same", "constant"])
    def test_transform_as_inverse(self, dim, N, padding):
        # test equivalence between transforms within an env and within a rb
        in_keys = ["observation", ("next", "observation")]
        rollout_length = 10
        cat_frames = CatFrames(
            N=N, in_keys=in_keys, dim=dim, padding=padding, as_inverse=True
        )

        env1 = TransformedEnv(
            ContinuousActionVecMockEnv(),
        )
        env2 = TransformedEnv(
            ContinuousActionVecMockEnv(),
            CatFrames(N=N, in_keys=in_keys, dim=dim, padding=padding, as_inverse=True),
        )
        obs_dim = env1.observation_spec["observation_orig"].shape[0]
        td = env1.rollout(rollout_length)

        transformed_td = cat_frames._inv_call(td)
        assert transformed_td.get(in_keys[0]).shape == (rollout_length, obs_dim * N)
        assert transformed_td.get(in_keys[1]).shape == (rollout_length, obs_dim * N)
        with pytest.raises(
            Exception,
            match="CatFrames as inverse is not supported as a transform for environments, only for replay buffers.",
        ):
            env2.rollout(rollout_length)

    def test_catframes_transform_observation_spec(self):
        N = 4
        key1 = "first key"
        key2 = "second key"
        keys = [key1, key2]
        cat_frames = CatFrames(
            N=N,
            in_keys=keys,
            dim=-3,
        )
        mins = [0, 0.5]
        maxes = [0.5, 1]
        observation_spec = Composite(
            {
                key: Bounded(space_min, space_max, (1, 3, 3), dtype=torch.double)
                for key, space_min, space_max in zip(keys, mins, maxes)
            }
        )

        result = cat_frames.transform_observation_spec(observation_spec.clone())
        observation_spec = Composite(
            {
                key: Bounded(space_min, space_max, (1, 3, 3), dtype=torch.double)
                for key, space_min, space_max in zip(keys, mins, maxes)
            }
        )

        final_spec = result[key2]
        assert final_spec.shape[0] == N
        for key in keys:
            for i in range(N):
                assert torch.equal(
                    result[key].space.high[i], observation_spec[key].space.high[0]
                )
                assert torch.equal(
                    result[key].space.low[i], observation_spec[key].space.low[0]
                )

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("batch_size", [(), (1,), (1, 2)])
    @pytest.mark.parametrize("d", range(1, 4))
    @pytest.mark.parametrize("dim", [-3, -2, 1])
    @pytest.mark.parametrize("N", [2, 4])
    def test_transform_no_env(self, device, d, batch_size, dim, N):
        key1 = "first key"
        key2 = ("second", "key")
        keys = [key1, key2]
        extra_d = (3,) * (-dim - 1)
        key1_tensor = torch.ones(*batch_size, d, *extra_d, device=device) * 2
        key2_tensor = torch.ones(*batch_size, d, *extra_d, device=device)
        key_tensors = [key1_tensor, key2_tensor]
        td = TensorDict(dict(zip(keys, key_tensors)), batch_size, device=device)
        if dim > 0:
            with pytest.raises(
                ValueError, match="dim must be < 0 to accommodate for tensordict"
            ):
                cat_frames = CatFrames(N=N, in_keys=keys, dim=dim)
            return
        cat_frames = CatFrames(N=N, in_keys=keys, dim=dim)

        tdclone = cat_frames._call(td.clone())
        latest_frame = tdclone.get(key2)

        assert latest_frame.shape[dim] == N * d
        slices = (slice(None),) * (-dim - 1)
        index1 = (Ellipsis, slice(None, -d), *slices)
        index2 = (Ellipsis, slice(-d, None), *slices)
        assert (latest_frame[index1] == 0).all()
        assert (latest_frame[index2] == 1).all()
        v1 = latest_frame[index1]

        tdclone = cat_frames._call(td.clone())
        latest_frame = tdclone.get(key2)

        assert latest_frame.shape[dim] == N * d
        index1 = (Ellipsis, slice(None, -2 * d), *slices)
        index2 = (Ellipsis, slice(-2 * d, None), *slices)
        assert (latest_frame[index1] == 0).all()
        assert (latest_frame[index2] == 1).all()
        v2 = latest_frame[index1]

        # we don't want the same tensor to be returned twice, but they're all copies of the same buffer
        assert v1 is not v2

    @pytest.mark.skipif(not _has_gym, reason="gym required for this test")
    @pytest.mark.parametrize("padding", ["constant", "same"])
    @pytest.mark.parametrize("envtype", ["gym", "conv"])
    def test_tranform_offline_against_online(self, padding, envtype):
        torch.manual_seed(0)
        key = "observation" if envtype == "gym" else "pixels"
        env = SerialEnv(
            3,
            lambda: TransformedEnv(
                GymEnv("CartPole-v1")
                if envtype == "gym"
                else DiscreteActionConvMockEnv(),
                CatFrames(
                    dim=-3 if envtype == "conv" else -1,
                    N=5,
                    in_keys=[key],
                    out_keys=[f"{key}_cat"],
                    padding=padding,
                ),
            ),
        )
        env.set_seed(0)

        r = env.rollout(100, break_when_any_done=False)

        c = CatFrames(
            dim=-3 if envtype == "conv" else -1,
            N=5,
            in_keys=[key, ("next", key)],
            out_keys=[f"{key}_cat2", ("next", f"{key}_cat2")],
            padding=padding,
        )

        r2 = c(r)

        torch.testing.assert_close(r2[f"{key}_cat2"], r2[f"{key}_cat"])
        torch.testing.assert_close(r2["next", f"{key}_cat2"], r2["next", f"{key}_cat"])

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("batch_size", [(), (1,), (1, 2)])
    @pytest.mark.parametrize("d", range(2, 3))
    @pytest.mark.parametrize("dim", [-3])
    @pytest.mark.parametrize("N", [2, 4])
    def test_transform_compose(self, device, d, batch_size, dim, N):
        key1 = "first key"
        key2 = "second key"
        keys = [key1, key2]
        extra_d = (3,) * (-dim - 1)
        key1_tensor = torch.ones(*batch_size, d, *extra_d, device=device) * 2
        key2_tensor = torch.ones(*batch_size, d, *extra_d, device=device)
        key_tensors = [key1_tensor, key2_tensor]
        td = TensorDict(dict(zip(keys, key_tensors)), batch_size, device=device)
        cat_frames = Compose(CatFrames(N=N, in_keys=keys, dim=dim))

        tdclone = cat_frames._call(td.clone())
        latest_frame = tdclone.get(key2)

        assert latest_frame.shape[dim] == N * d
        slices = (slice(None),) * (-dim - 1)
        index1 = (Ellipsis, slice(None, -d), *slices)
        index2 = (Ellipsis, slice(-d, None), *slices)
        assert (latest_frame[index1] == 0).all()
        assert (latest_frame[index2] == 1).all()
        v1 = latest_frame[index1]

        tdclone = cat_frames._call(td.clone())
        latest_frame = tdclone.get(key2)

        assert latest_frame.shape[dim] == N * d
        index1 = (Ellipsis, slice(None, -2 * d), *slices)
        index2 = (Ellipsis, slice(-2 * d, None), *slices)
        assert (latest_frame[index1] == 0).all()
        assert (latest_frame[index2] == 1).all()
        v2 = latest_frame[index1]

        # we don't want the same tensor to be returned twice, but they're all copies of the same buffer
        assert v1 is not v2

    @pytest.mark.parametrize("device", get_default_devices())
    def test_catframes_reset(self, device):
        key1 = "first key"
        key2 = "second key"
        N = 4
        keys = [key1, key2]
        key1_tensor = torch.randn(1, 1, 3, 3, device=device)
        key2_tensor = torch.randn(1, 1, 3, 3, device=device)
        key_tensors = [key1_tensor, key2_tensor]
        td = TensorDict(dict(zip(keys, key_tensors)), [1], device=device)
        cat_frames = CatFrames(N=N, in_keys=keys, dim=-3, reset_key="_reset")

        cat_frames._call(td.clone())
        buffer = getattr(cat_frames, f"_cat_buffers_{key1}")

        tdc = td.clone()
        cat_frames._reset(tdc, tdc)

        # assert tdc is passed_back_td
        # assert (buffer == 0).all()
        #
        # _ = cat_frames._call(tdc)
        assert (buffer != 0).all()

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for CatFrames")

    @pytest.mark.parametrize("padding_value", [2, 0.5, -1])
    def test_constant_padding(self, padding_value):
        key1 = "first_key"
        N = 4
        key1_tensor = torch.zeros((1, 1))
        td = TensorDict({key1: key1_tensor}, [1])
        cat_frames = CatFrames(
            N=N,
            in_keys=key1,
            out_keys="cat_" + key1,
            dim=-1,
            padding="constant",
            padding_value=padding_value,
        )

        cat_td = cat_frames._call(td.clone())
        assert (cat_td.get("cat_first_key") == padding_value).sum() == N - 1
        cat_td = cat_frames._call(cat_td)
        assert (cat_td.get("cat_first_key") == padding_value).sum() == N - 2
        cat_td = cat_frames._call(cat_td)
        assert (cat_td.get("cat_first_key") == padding_value).sum() == N - 3
        cat_td = cat_frames._call(cat_td)
        assert (cat_td.get("cat_first_key") == padding_value).sum() == N - 4


@pytest.mark.skipif(not _has_tv, reason="no torchvision")
class TestCrop(TransformBase):
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("h", [None, 21])
    @pytest.mark.parametrize(
        "keys", [["observation", ("some_other", "nested_key")], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_no_env(self, keys, h, nchannels, batch, device):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, nchannels, 16, 16, device=device)
        crop = Crop(w=20, h=h, in_keys=keys)
        if h is None:
            h = 20
        td = TensorDict(
            {
                key: torch.randn(*batch, nchannels, 16, 16, device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        crop(td)
        for key in keys:
            assert td.get(key).shape[-2:] == torch.Size([20, h])
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = Bounded(-1, 1, (nchannels, 16, 16))
            observation_spec = crop.transform_observation_spec(observation_spec.clone())
            assert observation_spec.shape == torch.Size([nchannels, 20, h])
        else:
            observation_spec = Composite(
                {key: Bounded(-1, 1, (nchannels, 16, 16)) for key in keys}
            )
            observation_spec = crop.transform_observation_spec(observation_spec.clone())
            for key in keys:
                assert observation_spec[key].shape == torch.Size([nchannels, 20, h])

    @pytest.mark.parametrize("nchannels", [3])
    @pytest.mark.parametrize("batch", [[2]])
    @pytest.mark.parametrize("h", [None])
    @pytest.mark.parametrize("keys", [["observation_pixels"]])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_model(self, keys, h, nchannels, batch, device):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, nchannels, 16, 16, device=device)
        crop = Crop(w=20, h=h, in_keys=keys)
        if h is None:
            h = 20
        td = TensorDict(
            {
                key: torch.randn(*batch, nchannels, 16, 16, device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        model = nn.Sequential(crop, nn.Identity())
        model(td)
        for key in keys:
            assert td.get(key).shape[-2:] == torch.Size([20, h])
        assert (td.get("dont touch") == dont_touch).all()

    @pytest.mark.parametrize("nchannels", [3])
    @pytest.mark.parametrize("batch", [[2]])
    @pytest.mark.parametrize("h", [None])
    @pytest.mark.parametrize("keys", [["observation_pixels"]])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_compose(self, keys, h, nchannels, batch, device):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, nchannels, 16, 16, device=device)
        crop = Crop(w=20, h=h, in_keys=keys)
        if h is None:
            h = 20
        td = TensorDict(
            {
                key: torch.randn(*batch, nchannels, 16, 16, device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        model = Compose(crop)
        tdc = model(td.clone())
        for key in keys:
            assert tdc.get(key).shape[-2:] == torch.Size([20, h])
        assert (tdc.get("dont touch") == dont_touch).all()
        tdc = model._call(td.clone())
        for key in keys:
            assert tdc.get(key).shape[-2:] == torch.Size([20, h])
        assert (tdc.get("dont touch") == dont_touch).all()

    @pytest.mark.parametrize("nchannels", [3])
    @pytest.mark.parametrize("batch", [[2]])
    @pytest.mark.parametrize("h", [None])
    @pytest.mark.parametrize("keys", [["observation_pixels"]])
    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(
        self,
        rbclass,
        keys,
        h,
        nchannels,
        batch,
    ):
        torch.manual_seed(0)
        dont_touch = torch.randn(
            *batch,
            nchannels,
            16,
            16,
        )
        crop = Crop(w=20, h=h, in_keys=keys)
        if h is None:
            h = 20
        td = TensorDict(
            {
                key: torch.randn(
                    *batch,
                    nchannels,
                    16,
                    16,
                )
                for key in keys
            },
            batch,
        )
        td.set("dont touch", dont_touch.clone())
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(crop)
        rb.extend(td)
        td = rb.sample(10)
        for key in keys:
            assert td.get(key).shape[-2:] == torch.Size([20, h])

    def test_single_trans_env_check(self):
        keys = ["pixels"]
        ct = Compose(ToTensorImage(), Crop(w=20, h=20, in_keys=keys))
        env = TransformedEnv(DiscreteActionConvMockEnvNumpy(), ct)
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        keys = ["pixels"]

        def make_env():
            ct = Compose(ToTensorImage(), Crop(w=20, h=20, in_keys=keys))
            return TransformedEnv(DiscreteActionConvMockEnvNumpy(), ct)

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self):
        keys = ["pixels"]

        def make_env():
            ct = Compose(ToTensorImage(), Crop(w=20, h=20, in_keys=keys))
            return TransformedEnv(DiscreteActionConvMockEnvNumpy(), ct)

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        keys = ["pixels"]
        ct = Compose(ToTensorImage(), Crop(w=20, h=20, in_keys=keys))
        env = TransformedEnv(SerialEnv(2, DiscreteActionConvMockEnvNumpy), ct)
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        keys = ["pixels"]
        ct = Compose(ToTensorImage(), Crop(w=20, h=20, in_keys=keys))
        env = TransformedEnv(ParallelEnv(2, DiscreteActionConvMockEnvNumpy), ct)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.skipif(not _has_gym, reason="No Gym detected")
    @pytest.mark.parametrize("out_key", [None, ["outkey"], [("out", "key")]])
    def test_transform_env(self, out_key):
        if not _has_ale:
            pytest.skip("ALE not available (missing ale_py); skipping Atari gym test.")
        keys = ["pixels"]
        ct = Compose(ToTensorImage(), Crop(out_keys=out_key, w=20, h=20, in_keys=keys))
        env = TransformedEnv(GymEnv(PONG_VERSIONED()), ct)
        td = env.reset()
        if out_key is None:
            assert td["pixels"].shape == torch.Size([3, 20, 20])
        else:
            assert td[out_key[0]].shape == torch.Size([3, 20, 20])
        check_env_specs(env)

    def test_transform_inverse(self):
        raise pytest.skip("Crop does not have an inverse method.")


@pytest.mark.skipif(not _has_tv, reason="no torchvision")
class TestCenterCrop(TransformBase):
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("h", [None, 21])
    @pytest.mark.parametrize(
        "keys", [["observation", ("some_other", "nested_key")], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_no_env(self, keys, h, nchannels, batch, device):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, nchannels, 16, 16, device=device)
        cc = CenterCrop(w=20, h=h, in_keys=keys)
        if h is None:
            h = 20
        td = TensorDict(
            {
                key: torch.randn(*batch, nchannels, 16, 16, device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        cc(td)
        for key in keys:
            assert td.get(key).shape[-2:] == torch.Size([20, h])
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = Bounded(-1, 1, (nchannels, 16, 16))
            observation_spec = cc.transform_observation_spec(observation_spec.clone())
            assert observation_spec.shape == torch.Size([nchannels, 20, h])
        else:
            observation_spec = Composite(
                {key: Bounded(-1, 1, (nchannels, 16, 16)) for key in keys}
            )
            observation_spec = cc.transform_observation_spec(observation_spec.clone())
            for key in keys:
                assert observation_spec[key].shape == torch.Size([nchannels, 20, h])

    @pytest.mark.parametrize("nchannels", [3])
    @pytest.mark.parametrize("batch", [[2]])
    @pytest.mark.parametrize("h", [None])
    @pytest.mark.parametrize("keys", [["observation_pixels"]])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_model(self, keys, h, nchannels, batch, device):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, nchannels, 16, 16, device=device)
        cc = CenterCrop(w=20, h=h, in_keys=keys)
        if h is None:
            h = 20
        td = TensorDict(
            {
                key: torch.randn(*batch, nchannels, 16, 16, device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        model = nn.Sequential(cc, nn.Identity())
        model(td)
        for key in keys:
            assert td.get(key).shape[-2:] == torch.Size([20, h])
        assert (td.get("dont touch") == dont_touch).all()

    @pytest.mark.parametrize("nchannels", [3])
    @pytest.mark.parametrize("batch", [[2]])
    @pytest.mark.parametrize("h", [None])
    @pytest.mark.parametrize("keys", [["observation_pixels"]])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_compose(self, keys, h, nchannels, batch, device):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, nchannels, 16, 16, device=device)
        cc = CenterCrop(w=20, h=h, in_keys=keys)
        if h is None:
            h = 20
        td = TensorDict(
            {
                key: torch.randn(*batch, nchannels, 16, 16, device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        model = Compose(cc)
        tdc = model(td.clone())
        for key in keys:
            assert tdc.get(key).shape[-2:] == torch.Size([20, h])
        assert (tdc.get("dont touch") == dont_touch).all()
        tdc = model._call(td.clone())
        for key in keys:
            assert tdc.get(key).shape[-2:] == torch.Size([20, h])
        assert (tdc.get("dont touch") == dont_touch).all()

    @pytest.mark.parametrize("nchannels", [3])
    @pytest.mark.parametrize("batch", [[2]])
    @pytest.mark.parametrize("h", [None])
    @pytest.mark.parametrize("keys", [["observation_pixels"]])
    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(
        self,
        rbclass,
        keys,
        h,
        nchannels,
        batch,
    ):
        torch.manual_seed(0)
        dont_touch = torch.randn(
            *batch,
            nchannels,
            16,
            16,
        )
        cc = CenterCrop(w=20, h=h, in_keys=keys)
        if h is None:
            h = 20
        td = TensorDict(
            {
                key: torch.randn(
                    *batch,
                    nchannels,
                    16,
                    16,
                )
                for key in keys
            },
            batch,
        )
        td.set("dont touch", dont_touch.clone())
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(cc)
        rb.extend(td)
        td = rb.sample(10)
        for key in keys:
            assert td.get(key).shape[-2:] == torch.Size([20, h])

    def test_single_trans_env_check(self):
        keys = ["pixels"]
        ct = Compose(ToTensorImage(), CenterCrop(w=20, h=20, in_keys=keys))
        env = TransformedEnv(DiscreteActionConvMockEnvNumpy(), ct)
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        keys = ["pixels"]

        def make_env():
            ct = Compose(ToTensorImage(), CenterCrop(w=20, h=20, in_keys=keys))
            return TransformedEnv(DiscreteActionConvMockEnvNumpy(), ct)

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self):
        keys = ["pixels"]

        def make_env():
            ct = Compose(ToTensorImage(), CenterCrop(w=20, h=20, in_keys=keys))
            return TransformedEnv(DiscreteActionConvMockEnvNumpy(), ct)

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        keys = ["pixels"]
        ct = Compose(ToTensorImage(), CenterCrop(w=20, h=20, in_keys=keys))
        env = TransformedEnv(SerialEnv(2, DiscreteActionConvMockEnvNumpy), ct)
        check_env_specs(env)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        keys = ["pixels"]
        ct = Compose(ToTensorImage(), CenterCrop(w=20, h=20, in_keys=keys))
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, DiscreteActionConvMockEnvNumpy), ct
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.skipif(not _has_gym, reason="No Gym detected")
    @pytest.mark.parametrize("out_key", [None, ["outkey"], [("out", "key")]])
    def test_transform_env(self, out_key):
        if not _has_ale:
            pytest.skip("ALE not available (missing ale_py); skipping Atari gym test.")
        keys = ["pixels"]
        ct = Compose(
            ToTensorImage(), CenterCrop(out_keys=out_key, w=20, h=20, in_keys=keys)
        )
        env = TransformedEnv(GymEnv(PONG_VERSIONED()), ct)
        td = env.reset()
        if out_key is None:
            assert td["pixels"].shape == torch.Size([3, 20, 20])
        else:
            assert td[out_key[0]].shape == torch.Size([3, 20, 20])
        check_env_specs(env)

    def test_transform_inverse(self):
        raise pytest.skip("CenterCrop does not have an inverse method.")


class TestFlattenObservation(TransformBase):
    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_single_trans_env_check(self, out_keys):
        env = TransformedEnv(
            DiscreteActionConvMockEnvNumpy(),
            FlattenObservation(-3, -1, out_keys=out_keys),
        )
        check_env_specs(env)
        if out_keys:
            assert out_keys[0] in env.reset().keys()

    def test_serial_trans_env_check(self):
        def make_env():
            env = TransformedEnv(
                DiscreteActionConvMockEnvNumpy(), FlattenObservation(-3, -1)
            )
            return env

        SerialEnv(2, make_env).check_env_specs()

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            env = TransformedEnv(
                DiscreteActionConvMockEnvNumpy(), FlattenObservation(-3, -1)
            )
            return env

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, DiscreteActionConvMockEnvNumpy),
            FlattenObservation(
                -3,
                -1,
            ),
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, DiscreteActionConvMockEnvNumpy),
            FlattenObservation(
                -3,
                -1,
            ),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.skipif(not _has_tv, reason="no torchvision")
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("size", [[], [4]])
    @pytest.mark.parametrize(
        "keys", [["observation", ("some_other", "nested_key")], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_no_env(self, keys, size, nchannels, batch, device):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, *size, nchannels, 16, 16, device=device)
        start_dim = -3 - len(size)
        flatten = FlattenObservation(start_dim, -3, in_keys=keys)
        td = TensorDict(
            {
                key: torch.randn(*batch, *size, nchannels, 16, 16, device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        flatten(td)
        expected_size = prod(size + [nchannels])
        for key in keys:
            assert td.get(key).shape[-3] == expected_size
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = Bounded(-1, 1, (*size, nchannels, 16, 16))
            observation_spec = flatten.transform_observation_spec(
                observation_spec.clone()
            )
            assert observation_spec.shape[-3] == expected_size
        else:
            observation_spec = Composite(
                {key: Bounded(-1, 1, (*size, nchannels, 16, 16)) for key in keys}
            )
            observation_spec = flatten.transform_observation_spec(
                observation_spec.clone()
            )
            for key in keys:
                assert observation_spec[key].shape[-3] == expected_size

    @pytest.mark.skipif(not _has_tv, reason="no torchvision")
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("size", [[], [4]])
    @pytest.mark.parametrize(
        "keys", [["observation", "some_other_key"], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_compose(self, keys, size, nchannels, batch, device):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, *size, nchannels, 16, 16, device=device)
        start_dim = -3 - len(size)
        flatten = Compose(FlattenObservation(start_dim, -3, in_keys=keys))
        td = TensorDict(
            {
                key: torch.randn(*batch, *size, nchannels, 16, 16, device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        flatten(td)
        expected_size = prod(size + [nchannels])
        for key in keys:
            assert td.get(key).shape[-3] == expected_size
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = Bounded(-1, 1, (*size, nchannels, 16, 16))
            observation_spec = flatten.transform_observation_spec(
                observation_spec.clone()
            )
            assert observation_spec.shape[-3] == expected_size
        else:
            observation_spec = Composite(
                {key: Bounded(-1, 1, (*size, nchannels, 16, 16)) for key in keys}
            )
            observation_spec = flatten.transform_observation_spec(
                observation_spec.clone()
            )
            for key in keys:
                assert observation_spec[key].shape[-3] == expected_size

    @pytest.mark.skipif(not _has_gym, reason="No gym")
    @pytest.mark.parametrize(
        "out_keys", [None, ["stuff"], [("some_other", "nested_key")]]
    )
    def test_transform_env(self, out_keys):
        if not _has_ale:
            pytest.skip("ALE not available (missing ale_py); skipping Atari gym test.")
        env = TransformedEnv(
            GymEnv(PONG_VERSIONED()), FlattenObservation(-3, -1, out_keys=out_keys)
        )
        check_env_specs(env)
        if out_keys:
            assert out_keys[0] in env.reset().keys(True, True)
            assert env.rollout(3)[out_keys[0]].ndimension() == 2
        else:
            assert env.rollout(3)["pixels"].ndimension() == 2

    @pytest.mark.skipif(not _has_gym, reason="No gym")
    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_transform_model(self, out_keys):
        t = FlattenObservation(-3, -1, out_keys=out_keys)
        td = TensorDict({"pixels": torch.randint(255, (10, 10, 3))}, [])
        module = nn.Sequential(t, nn.Identity())
        if out_keys:
            assert module(td)[out_keys[0]].ndimension() == 1
        else:
            assert module(td)["pixels"].ndimension() == 1

    @pytest.mark.skipif(not _has_gym, reason="No gym")
    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, out_keys, rbclass):
        t = FlattenObservation(-3, -1, out_keys=out_keys)
        td = TensorDict({"pixels": torch.randint(255, (10, 10, 3))}, []).expand(10)
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(t)
        rb.extend(td)
        td = rb.sample(2)
        if out_keys:
            assert td[out_keys[0]].ndimension() == 2
        else:
            assert td["pixels"].ndimension() == 2

    def test_transform_inverse(self):
        raise pytest.skip("No inverse method for FlattenObservation (yet).")


class TestGrayScale(TransformBase):
    @pytest.mark.skipif(not _has_tv, reason="no torchvision")
    @pytest.mark.parametrize(
        "keys",
        [
            [("next", "observation"), ("some_other", "nested_key")],
            [("next", "observation_pixels")],
        ],
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_no_env(self, keys, device):
        torch.manual_seed(0)
        nchannels = 3
        gs = GrayScale(in_keys=keys)
        dont_touch = torch.randn(1, nchannels, 16, 16, device=device)
        td = TensorDict(
            {key: torch.randn(1, nchannels, 16, 16, device=device) for key in keys},
            [1],
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        gs(td)
        for key in keys:
            assert td.get(key).shape[-3] == 1
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = Bounded(-1, 1, (nchannels, 16, 16))
            observation_spec = gs.transform_observation_spec(observation_spec.clone())
            assert observation_spec.shape == torch.Size([1, 16, 16])
        else:
            observation_spec = Composite(
                {key: Bounded(-1, 1, (nchannels, 16, 16)) for key in keys}
            )
            observation_spec = gs.transform_observation_spec(observation_spec.clone())
            for key in keys:
                assert observation_spec[key].shape == torch.Size([1, 16, 16])

    @pytest.mark.skipif(not _has_tv, reason="no torchvision")
    @pytest.mark.parametrize(
        "keys",
        [
            [("next", "observation"), ("some_other", "nested_key")],
            [("next", "observation_pixels")],
        ],
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_compose(self, keys, device):
        torch.manual_seed(0)
        nchannels = 3
        gs = Compose(GrayScale(in_keys=keys))
        dont_touch = torch.randn(1, nchannels, 16, 16, device=device)
        td = TensorDict(
            {key: torch.randn(1, nchannels, 16, 16, device=device) for key in keys},
            [1],
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        gs(td)
        for key in keys:
            assert td.get(key).shape[-3] == 1
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = Bounded(-1, 1, (nchannels, 16, 16))
            observation_spec = gs.transform_observation_spec(observation_spec.clone())
            assert observation_spec.shape == torch.Size([1, 16, 16])
        else:
            observation_spec = Composite(
                {key: Bounded(-1, 1, (nchannels, 16, 16)) for key in keys}
            )
            observation_spec = gs.transform_observation_spec(observation_spec.clone())
            for key in keys:
                assert observation_spec[key].shape == torch.Size([1, 16, 16])

    @pytest.mark.parametrize(
        "out_keys", [None, ["stuff"], [("some_other", "nested_key")]]
    )
    def test_single_trans_env_check(self, out_keys):
        env = TransformedEnv(
            DiscreteActionConvMockEnvNumpy(),
            Compose(ToTensorImage(), GrayScale(out_keys=out_keys)),
        )
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        out_keys = None

        def make_env():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy(),
                Compose(ToTensorImage(), GrayScale(out_keys=out_keys)),
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        out_keys = None

        def make_env():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy(),
                Compose(ToTensorImage(), GrayScale(out_keys=out_keys)),
            )

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        out_keys = None
        env = TransformedEnv(
            SerialEnv(2, DiscreteActionConvMockEnvNumpy),
            Compose(ToTensorImage(), GrayScale(out_keys=out_keys)),
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        out_keys = None
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, DiscreteActionConvMockEnvNumpy),
            Compose(ToTensorImage(), GrayScale(out_keys=out_keys)),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_transform_env(self, out_keys):
        env = TransformedEnv(
            DiscreteActionConvMockEnvNumpy(),
            Compose(ToTensorImage(), GrayScale(out_keys=out_keys)),
        )
        r = env.rollout(3)
        if out_keys:
            assert "pixels" in r.keys()
            assert "stuff" in r.keys()
            assert r["pixels"].shape[-3] == 3
            assert r["stuff"].shape[-3] == 1
        else:
            assert "pixels" in r.keys()
            assert "stuff" not in r.keys()
            assert r["pixels"].shape[-3] == 1

    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_transform_model(self, out_keys):
        td = TensorDict({"pixels": torch.rand(3, 12, 12)}, []).expand(3)
        model = nn.Sequential(GrayScale(out_keys=out_keys), nn.Identity())
        r = model(td)
        if out_keys:
            assert "pixels" in r.keys()
            assert "stuff" in r.keys()
            assert r["pixels"].shape[-3] == 3
            assert r["stuff"].shape[-3] == 1
        else:
            assert "pixels" in r.keys()
            assert "stuff" not in r.keys()
            assert r["pixels"].shape[-3] == 1

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_transform_rb(self, out_keys, rbclass):
        td = TensorDict({"pixels": torch.rand(3, 12, 12)}, []).expand(3)
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(GrayScale(out_keys=out_keys))
        rb.extend(td)
        r = rb.sample(3)
        if out_keys:
            assert "pixels" in r.keys()
            assert "stuff" in r.keys()
            assert r["pixels"].shape[-3] == 3
            assert r["stuff"].shape[-3] == 1
        else:
            assert "pixels" in r.keys()
            assert "stuff" not in r.keys()
            assert r["pixels"].shape[-3] == 1

    def test_transform_inverse(self):
        raise pytest.skip("No inversee for grayscale")


class TestObservationNorm(TransformBase):
    @pytest.mark.parametrize(
        "out_keys", [None, ["stuff"], [("some_other", "nested_key")]]
    )
    def test_single_trans_env_check(
        self,
        out_keys,
    ):
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            ObservationNorm(
                loc=torch.zeros(7),
                scale=1.0,
                in_keys=["observation"],
                out_keys=out_keys,
            ),
        )
        check_env_specs(env)

    def test_serial_trans_env_check(
        self,
    ):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                ObservationNorm(
                    loc=torch.zeros(7),
                    in_keys=["observation"],
                    scale=1.0,
                ),
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                ObservationNorm(
                    loc=torch.zeros(7),
                    in_keys=["observation"],
                    scale=1.0,
                ),
            )

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(
        self,
    ):
        env = TransformedEnv(
            SerialEnv(2, ContinuousActionVecMockEnv),
            ObservationNorm(
                loc=torch.zeros(7),
                in_keys=["observation"],
                scale=1.0,
            ),
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, ContinuousActionVecMockEnv),
            ObservationNorm(
                loc=torch.zeros(7),
                in_keys=["observation"],
                scale=1.0,
            ),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize("standard_normal", [True, False])
    @pytest.mark.parametrize("in_key", ["observation", ("some_other", "observation")])
    @pytest.mark.parametrize(
        "out_keys", [None, ["stuff"], [("some_other", "nested_key")]]
    )
    def test_transform_no_env(self, out_keys, standard_normal, in_key):
        t = ObservationNorm(in_keys=[in_key], out_keys=out_keys)
        # test that init fails
        with pytest.raises(
            RuntimeError,
            match="Cannot initialize the transform if parent env is not defined",
        ):
            t.init_stats(num_iter=5)
        t = ObservationNorm(
            loc=torch.ones(7),
            scale=0.5,
            in_keys=[in_key],
            out_keys=out_keys,
            standard_normal=standard_normal,
        )
        obs = torch.randn(7)
        td = TensorDict({in_key: obs}, [])
        t(td)
        if out_keys:
            assert out_keys[0] in td.keys(True, True)
            obs_tr = td[out_keys[0]]
        else:
            obs_tr = td[in_key]
        if standard_normal:
            assert torch.allclose((obs - 1) / 0.5, obs_tr)
        else:
            assert torch.allclose(0.5 * obs + 1, obs_tr)

    @pytest.mark.parametrize("standard_normal", [True, False])
    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_transform_compose(self, out_keys, standard_normal):
        t = Compose(ObservationNorm(in_keys=["observation"], out_keys=out_keys))
        # test that init fails
        with pytest.raises(
            RuntimeError,
            match="Cannot initialize the transform if parent env is not defined",
        ):
            t[0].init_stats(num_iter=5)
        t = Compose(
            ObservationNorm(
                loc=torch.ones(7),
                scale=0.5,
                in_keys=["observation"],
                out_keys=out_keys,
                standard_normal=standard_normal,
            )
        )
        obs = torch.randn(7)
        td = TensorDict({"observation": obs}, [])
        t(td)
        if out_keys:
            assert out_keys[0] in td.keys()
            obs_tr = td[out_keys[0]]
        else:
            obs_tr = td["observation"]
        if standard_normal:
            assert torch.allclose((obs - 1) / 0.5, obs_tr)
        else:
            assert torch.allclose(0.5 * obs + 1, obs_tr)

    @pytest.mark.parametrize("standard_normal", [True, False])
    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_transform_env(self, out_keys, standard_normal):
        if standard_normal:
            scale = 1_000_000
        else:
            scale = 0.0
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            ObservationNorm(
                loc=0.0,
                scale=scale,
                in_keys=["observation"],
                out_keys=out_keys,
                standard_normal=standard_normal,
            ),
        )
        if out_keys:
            assert out_keys[0] in env.reset().keys()
            obs = env.rollout(3)[out_keys[0]]
        else:
            obs = env.rollout(3)["observation"]

        assert (abs(obs) < 1e-2).all()

    @pytest.mark.parametrize("standard_normal", [True, False])
    def test_transform_env_clone(self, standard_normal):
        out_keys = ["stuff"]
        if standard_normal:
            scale = 1_000_000
        else:
            scale = 0.0
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            ObservationNorm(
                loc=0.0,
                scale=scale,
                in_keys=["observation"],
                out_keys=out_keys,
                standard_normal=standard_normal,
            ),
        )
        cloned = env.transform.clone()
        env.transform.loc += 1
        env.transform.scale += 1
        torch.testing.assert_close(
            env.transform.loc, torch.ones_like(env.transform.loc)
        )
        torch.testing.assert_close(
            env.transform.scale, scale + torch.ones_like(env.transform.scale)
        )
        assert env.transform.loc == cloned.loc
        assert env.transform.scale == cloned.scale

    def test_transform_model(self):
        standard_normal = True
        out_keys = ["stuff"]

        t = Compose(
            ObservationNorm(
                loc=torch.ones(7),
                scale=0.5,
                in_keys=["observation"],
                out_keys=out_keys,
                standard_normal=standard_normal,
            )
        )
        model = nn.Sequential(t, nn.Identity())
        obs = torch.randn(7)
        td = TensorDict({"observation": obs}, [])
        model(td)

        if out_keys:
            assert out_keys[0] in td.keys()
            obs_tr = td[out_keys[0]]
        else:
            obs_tr = td["observation"]
        if standard_normal:
            assert torch.allclose((obs - 1) / 0.5, obs_tr)
        else:
            assert torch.allclose(0.5 * obs + 1, obs_tr)

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        standard_normal = True
        out_keys = ["stuff"]

        t = Compose(
            ObservationNorm(
                loc=torch.ones(7),
                scale=0.5,
                in_keys=["observation"],
                out_keys=out_keys,
                standard_normal=standard_normal,
            )
        )
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(t)

        obs = torch.randn(7)
        td = TensorDict({"observation": obs}, []).expand(3)
        rb.extend(td)
        td = rb.sample(5)

        if out_keys:
            assert out_keys[0] in td.keys()
            obs_tr = td[out_keys[0]]
        else:
            obs_tr = td["observation"]
        if standard_normal:
            assert torch.allclose((obs - 1) / 0.5, obs_tr)
        else:
            assert torch.allclose(0.5 * obs + 1, obs_tr)

    @pytest.mark.skipif(not _has_gym, reason="No gym")
    @pytest.mark.parametrize("out_key_inv", ["action_inv", ("nested", "action_inv")])
    @pytest.mark.parametrize(
        "out_key", ["observation_out", ("nested", "observation_out")]
    )
    @pytest.mark.parametrize("compose", [False, True])
    def test_transform_inverse(self, out_key, out_key_inv, compose):
        standard_normal = True
        out_keys = [out_key]
        in_keys_inv = ["action"]
        out_keys_inv = [out_key_inv]
        t = ObservationNorm(
            loc=torch.ones(()),
            scale=0.5,
            in_keys=["observation"],
            out_keys=out_keys,
            # What the env asks for
            in_keys_inv=in_keys_inv,
            # What the outside world sees
            out_keys_inv=out_keys_inv,
            standard_normal=standard_normal,
        )
        if compose:
            t = Compose(t)
        base_env = GymEnv(PENDULUM_VERSIONED())
        env = TransformedEnv(base_env, t)
        assert out_keys_inv[0] in env.full_action_spec.keys(True, True)
        td = env.rollout(3)
        check_env_specs(env)
        env.set_seed(0)
        a, a_ = td[out_key_inv] * 0.5 + 1, t.inv(td)["action"]
        assert torch.allclose(a, a_), (a, a_)
        assert torch.allclose((td["observation"] - 1) / 0.5, td[out_key])

    @pytest.mark.parametrize("batch", [[], [1], [3, 2]])
    @pytest.mark.parametrize(
        "keys",
        [["next_observation", "some_other_key"], [("next", "observation_pixels")]],
    )
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("standard_normal", [True, False])
    @pytest.mark.parametrize(
        ["loc", "scale"],
        [
            (0, 1),
            (1, 2),
            (torch.ones(16, 16), torch.ones(1)),
            (torch.ones(1), torch.ones(16, 16)),
        ],
    )
    def test_observationnorm(
        self, batch, keys, device, nchannels, loc, scale, standard_normal
    ):
        torch.manual_seed(0)
        nchannels = 3
        if isinstance(loc, Tensor):
            loc = loc.to(device)
        if isinstance(scale, Tensor):
            scale = scale.to(device)
        on = ObservationNorm(loc, scale, in_keys=keys, standard_normal=standard_normal)
        dont_touch = torch.randn(1, nchannels, 16, 16, device=device)
        td = TensorDict(
            {key: torch.zeros(1, nchannels, 16, 16, device=device) for key in keys}, [1]
        )
        td.set("dont touch", dont_touch.clone())
        on(td)
        for key in keys:
            if standard_normal:
                assert (td.get(key) == -loc / scale).all()
            else:
                assert (td.get(key) == loc).all()
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = Bounded(0, 1, (nchannels, 16, 16), device=device)
            observation_spec = on.transform_observation_spec(observation_spec.clone())
            if standard_normal:
                assert (observation_spec.space.low == -loc / scale).all()
                assert (observation_spec.space.high == (1 - loc) / scale).all()
            else:
                assert (observation_spec.space.low == loc).all()
                assert (observation_spec.space.high == scale + loc).all()

        else:
            observation_spec = Composite(
                {key: Bounded(0, 1, (nchannels, 16, 16), device=device) for key in keys}
            )
            observation_spec = on.transform_observation_spec(observation_spec.clone())
            for key in keys:
                if standard_normal:
                    assert (observation_spec[key].space.low == -loc / scale).all()
                    assert (observation_spec[key].space.high == (1 - loc) / scale).all()
                else:
                    assert (observation_spec[key].space.low == loc).all()
                    assert (observation_spec[key].space.high == scale + loc).all()

    @pytest.mark.parametrize("keys", [["observation"], ["observation", "next_pixel"]])
    @pytest.mark.parametrize("size", [1, 3])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("standard_normal", [True, False])
    @pytest.mark.parametrize("parallel", [True, False])
    def test_observationnorm_init_stats(
        self, keys, size, device, standard_normal, parallel
    ):
        def make_env():
            base_env = ContinuousActionVecMockEnv(
                observation_spec=Composite(
                    observation=Bounded(low=1, high=1, shape=torch.Size([size])),
                    observation_orig=Bounded(low=1, high=1, shape=torch.Size([size])),
                ),
                action_spec=Bounded(low=1, high=1, shape=torch.Size((size,))),
                seed=0,
            )
            base_env.out_key = "observation"
            return base_env

        if parallel:
            base_env = SerialEnv(2, make_env)
            reduce_dim = (0, 1)
            cat_dim = 1
        else:
            base_env = make_env()
            reduce_dim = 0
            cat_dim = 0

        t_env = TransformedEnv(
            base_env,
            transform=ObservationNorm(in_keys=keys, standard_normal=standard_normal),
        )
        if len(keys) > 1:
            t_env.transform.init_stats(
                num_iter=11, key="observation", cat_dim=cat_dim, reduce_dim=reduce_dim
            )
        else:
            t_env.transform.init_stats(
                num_iter=11, reduce_dim=reduce_dim, cat_dim=cat_dim
            )
        batch_dims = len(t_env.batch_size)
        assert (
            t_env.transform.loc.shape
            == t_env.observation_spec["observation"].shape[batch_dims:]
        )
        assert (
            t_env.transform.scale.shape
            == t_env.observation_spec["observation"].shape[batch_dims:]
        )
        assert t_env.transform.loc.dtype == t_env.observation_spec["observation"].dtype
        assert (
            t_env.transform.loc.device == t_env.observation_spec["observation"].device
        )

    @pytest.mark.parametrize("keys", [["pixels"], ["pixels", "stuff"]])
    @pytest.mark.parametrize("size", [1, 3])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("standard_normal", [True, False])
    @pytest.mark.parametrize("parallel", [True, False])
    def test_observationnorm_init_stats_pixels(
        self, keys, size, device, standard_normal, parallel
    ):
        def make_env():
            base_env = DiscreteActionConvMockEnvNumpy(
                seed=0,
            )
            base_env.out_key = "pixels"
            return base_env

        if parallel:
            base_env = SerialEnv(2, make_env)
            reduce_dim = (0, 1, 3, 4)
            keep_dim = (3, 4)
            cat_dim = 1
        else:
            base_env = make_env()
            reduce_dim = (0, 2, 3)
            keep_dim = (2, 3)
            cat_dim = 0

        t_env = TransformedEnv(
            base_env,
            transform=ObservationNorm(in_keys=keys, standard_normal=standard_normal),
        )
        if len(keys) > 1:
            t_env.transform.init_stats(
                num_iter=11,
                key="pixels",
                cat_dim=cat_dim,
                reduce_dim=reduce_dim,
                keep_dims=keep_dim,
            )
        else:
            t_env.transform.init_stats(
                num_iter=11,
                reduce_dim=reduce_dim,
                cat_dim=cat_dim,
                keep_dims=keep_dim,
            )

        assert t_env.transform.loc.shape == torch.Size(
            [t_env.observation_spec["pixels"].shape[-3], 1, 1]
        )
        assert t_env.transform.scale.shape == torch.Size(
            [t_env.observation_spec["pixels"].shape[-3], 1, 1]
        )

    def test_observationnorm_stats_already_initialized_error(self):
        transform = ObservationNorm(in_keys=["next_observation"], loc=0, scale=1)

        with pytest.raises(RuntimeError, match="Loc/Scale are already initialized"):
            transform.init_stats(num_iter=11)

    def test_observationnorm_wrong_catdim(self):
        transform = ObservationNorm(in_keys=["next_observation"], loc=0, scale=1)

        with pytest.raises(
            ValueError, match="cat_dim must be part of or equal to reduce_dim"
        ):
            transform.init_stats(num_iter=11, cat_dim=1)

        with pytest.raises(
            ValueError, match="cat_dim must be part of or equal to reduce_dim"
        ):
            transform.init_stats(num_iter=11, cat_dim=2, reduce_dim=(0, 1))

        with pytest.raises(
            ValueError,
            match="cat_dim must be specified if reduce_dim is not an integer",
        ):
            transform.init_stats(num_iter=11, reduce_dim=(0, 1))

    def test_observationnorm_init_stats_multiple_keys_error(self):
        transform = ObservationNorm(in_keys=["next_observation", "next_pixels"])

        err_msg = "Transform has multiple in_keys but no specific key was passed as an argument"
        with pytest.raises(RuntimeError, match=err_msg):
            transform.init_stats(num_iter=11)

    def test_observationnorm_initialization_order_error(self):
        base_env = ContinuousActionVecMockEnv()
        t_env = TransformedEnv(base_env)

        transform1 = ObservationNorm(in_keys=["next_observation"])
        transform2 = ObservationNorm(in_keys=["next_observation"])
        t_env.append_transform(transform1)
        t_env.append_transform(transform2)

        err_msg = (
            "ObservationNorms need to be initialized in the right order."
            "Trying to initialize an ObservationNorm while a parent ObservationNorm transform is still uninitialized"
        )
        with pytest.raises(RuntimeError, match=err_msg):
            transform2.init_stats(num_iter=10, key="observation")

    def test_observationnorm_uninitialized_stats_error(self):
        transform = ObservationNorm(in_keys=["next_observation", "next_pixels"])

        err_msg = (
            "Loc/Scale have not been initialized. Either pass in values in the constructor "
            "or call the init_stats method"
        )
        with pytest.raises(RuntimeError, match=err_msg):
            transform._apply_transform(torch.Tensor([1]))


@pytest.mark.skipif(not _has_tv, reason="no torchvision")
class TestResize(TransformBase):
    @pytest.mark.parametrize("interpolation", ["bilinear", "bicubic"])
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize(
        "keys", [["observation", ("some_other", "nested_key")], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_no_env(self, interpolation, keys, nchannels, batch, device):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, nchannels, 16, 16, device=device)
        resize = Resize(w=20, h=21, interpolation=interpolation, in_keys=keys)
        td = TensorDict(
            {
                key: torch.randn(*batch, nchannels, 16, 16, device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        resize(td)
        for key in keys:
            assert td.get(key).shape[-2:] == torch.Size([20, 21])
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = Bounded(-1, 1, (nchannels, 16, 16))
            observation_spec = resize.transform_observation_spec(
                observation_spec.clone()
            )
            assert observation_spec.shape == torch.Size([nchannels, 20, 21])
        else:
            observation_spec = Composite(
                {key: Bounded(-1, 1, (nchannels, 16, 16)) for key in keys}
            )
            observation_spec = resize.transform_observation_spec(
                observation_spec.clone()
            )
            for key in keys:
                assert observation_spec[key].shape == torch.Size([nchannels, 20, 21])

    @pytest.mark.parametrize("interpolation", ["bilinear", "bicubic"])
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize(
        "keys", [["observation", "some_other_key"], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_compose(self, interpolation, keys, nchannels, batch, device):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, nchannels, 16, 16, device=device)
        resize = Compose(Resize(w=20, h=21, interpolation=interpolation, in_keys=keys))
        td = TensorDict(
            {
                key: torch.randn(*batch, nchannels, 16, 16, device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        resize(td)
        for key in keys:
            assert td.get(key).shape[-2:] == torch.Size([20, 21])
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = Bounded(-1, 1, (nchannels, 16, 16))
            observation_spec = resize.transform_observation_spec(
                observation_spec.clone()
            )
            assert observation_spec.shape == torch.Size([nchannels, 20, 21])
        else:
            observation_spec = Composite(
                {key: Bounded(-1, 1, (nchannels, 16, 16)) for key in keys}
            )
            observation_spec = resize.transform_observation_spec(
                observation_spec.clone()
            )
            for key in keys:
                assert observation_spec[key].shape == torch.Size([nchannels, 20, 21])

    def test_single_trans_env_check(self):
        env = TransformedEnv(
            DiscreteActionConvMockEnvNumpy(),
            Compose(ToTensorImage(), Resize(20, 21, in_keys=["pixels"])),
        )
        check_env_specs(env)
        assert "pixels" in env.observation_spec.keys()

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy(),
                Compose(ToTensorImage(), Resize(20, 21, in_keys=["pixels"])),
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy(),
                Compose(ToTensorImage(), Resize(20, 21, in_keys=["pixels"])),
            )

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, DiscreteActionConvMockEnvNumpy),
            Compose(ToTensorImage(), Resize(20, 21, in_keys=["pixels"])),
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, DiscreteActionConvMockEnvNumpy),
            Compose(ToTensorImage(), Resize(20, 21, in_keys=["pixels"])),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.skipif(not _has_gym, reason="No gym")
    @pytest.mark.parametrize("out_key", ["pixels", ("agents", "pixels")])
    def test_transform_env(self, out_key):
        if not _has_ale:
            pytest.skip("ALE not available (missing ale_py); skipping Atari gym test.")
        env = TransformedEnv(
            GymEnv(PONG_VERSIONED()),
            Compose(
                ToTensorImage(), Resize(20, 21, in_keys=["pixels"], out_keys=[out_key])
            ),
        )
        check_env_specs(env)
        td = env.rollout(3)
        assert td[out_key].shape[-3:] == torch.Size([3, 20, 21])

    def test_transform_model(self):
        module = nn.Sequential(Resize(20, 21, in_keys=["pixels"]), nn.Identity())
        td = TensorDict({"pixels": torch.randn(3, 32, 32)}, [])
        module(td)
        assert td["pixels"].shape == torch.Size([3, 20, 21])

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        t = Resize(20, 21, in_keys=["pixels"])
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(t)
        td = TensorDict({"pixels": torch.randn(3, 32, 32)}, []).expand(10)
        rb.extend(td)
        td = rb.sample(2)
        assert td["pixels"].shape[-3:] == torch.Size([3, 20, 21])

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for Resize")


class TestToTensorImage(TransformBase):
    @pytest.mark.parametrize("batch", [[], [1], [3, 2]])
    @pytest.mark.parametrize(
        "keys",
        [[("next", "observation"), "some_other_key"], [("next", "observation_pixels")]],
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_no_env(self, keys, batch, device):
        torch.manual_seed(0)
        nchannels = 3
        totensorimage = ToTensorImage(in_keys=keys)
        dont_touch = torch.randn(*batch, nchannels, 16, 16, device=device)
        td = TensorDict(
            {
                key: torch.randint(255, (*batch, 16, 16, 3), device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        totensorimage(td)
        for key in keys:
            assert td.get(key).shape[-3:] == torch.Size([3, 16, 16])
            assert td.get(key).device == device
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = Bounded(0, 255, (16, 16, 3), dtype=torch.uint8)
            observation_spec = totensorimage.transform_observation_spec(
                observation_spec.clone()
            )
            assert observation_spec.shape == torch.Size([3, 16, 16])
            assert (observation_spec.space.low == 0).all()
            assert (observation_spec.space.high == 1).all()
        else:
            observation_spec = Composite(
                {key: Bounded(0, 255, (16, 16, 3), dtype=torch.uint8) for key in keys}
            )
            observation_spec = totensorimage.transform_observation_spec(
                observation_spec.clone()
            )
            for key in keys:
                assert observation_spec[key].shape == torch.Size([3, 16, 16])
                assert (observation_spec[key].space.low == 0).all()
                assert (observation_spec[key].space.high == 1).all()

    @pytest.mark.parametrize("batch", [[], [1], [3, 2]])
    @pytest.mark.parametrize(
        "keys",
        [[("next", "observation"), "some_other_key"], [("next", "observation_pixels")]],
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_compose(self, keys, batch, device):
        torch.manual_seed(0)
        nchannels = 3
        totensorimage = Compose(ToTensorImage(in_keys=keys))
        dont_touch = torch.randn(*batch, nchannels, 16, 16, device=device)
        td = TensorDict(
            {
                key: torch.randint(255, (*batch, 16, 16, 3), device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        totensorimage(td)
        for key in keys:
            assert td.get(key).shape[-3:] == torch.Size([3, 16, 16])
            assert td.get(key).device == device
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = Bounded(0, 255, (16, 16, 3), dtype=torch.uint8)
            observation_spec = totensorimage.transform_observation_spec(
                observation_spec.clone()
            )
            assert observation_spec.shape == torch.Size([3, 16, 16])
            assert (observation_spec.space.low == 0).all()
            assert (observation_spec.space.high == 1).all()
        else:
            observation_spec = Composite(
                {key: Bounded(0, 255, (16, 16, 3), dtype=torch.uint8) for key in keys}
            )
            observation_spec = totensorimage.transform_observation_spec(
                observation_spec.clone()
            )
            for key in keys:
                assert observation_spec[key].shape == torch.Size([3, 16, 16])
                assert (observation_spec[key].space.low == 0).all()
                assert (observation_spec[key].space.high == 1).all()

    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_single_trans_env_check(self, out_keys):
        env = TransformedEnv(
            DiscreteActionConvMockEnvNumpy(),
            ToTensorImage(in_keys=["pixels"], out_keys=out_keys),
        )
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy(),
                ToTensorImage(in_keys=["pixels"], out_keys=None),
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy(),
                ToTensorImage(in_keys=["pixels"], out_keys=None),
            )

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, DiscreteActionConvMockEnvNumpy),
            ToTensorImage(in_keys=["pixels"], out_keys=None),
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, DiscreteActionConvMockEnvNumpy),
            ToTensorImage(in_keys=["pixels"], out_keys=None),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize("out_keys", [None, ["stuff"], [("nested", "stuff")]])
    @pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
    def test_transform_env(self, out_keys, default_dtype):
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(default_dtype)
        env = TransformedEnv(
            DiscreteActionConvMockEnvNumpy(),
            ToTensorImage(in_keys=["pixels"], out_keys=out_keys),
        )
        r = env.rollout(3)
        if out_keys is not None:
            assert out_keys[0] in r.keys(True, True)
            obs = r[out_keys[0]]
        else:
            obs = r["pixels"]
        assert obs.shape[-3] == 3
        assert obs.dtype is default_dtype
        torch.set_default_dtype(prev_dtype)

    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_transform_model(self, out_keys):
        t = ToTensorImage(in_keys=["pixels"], out_keys=out_keys)
        model = nn.Sequential(t, nn.Identity())
        td = TensorDict({"pixels": torch.randint(255, (21, 22, 3))}, [])
        model(td)
        if out_keys is not None:
            assert out_keys[0] in td.keys()
            obs = td[out_keys[0]]
        else:
            obs = td["pixels"]
        assert obs.shape[-3] == 3
        assert obs.dtype is torch.float32

    @pytest.mark.parametrize("from_int", [None, True, False])
    @pytest.mark.parametrize("default_dtype", [torch.float32, torch.uint8])
    def test_transform_scale(self, from_int, default_dtype):
        totensorimage = ToTensorImage(in_keys=["pixels"], from_int=from_int)
        fill_value = 150 if default_dtype == torch.uint8 else 0.5
        td = TensorDict(
            {"pixels": torch.full((21, 22, 3), fill_value, dtype=default_dtype)}, []
        )
        # Save whether or not the tensor is floating point before the transform changes it
        # to floating point type.
        is_floating_point = torch.is_floating_point(td["pixels"])
        totensorimage(td)

        if from_int is None:
            expected_pixel_value = (
                fill_value / 255 if not is_floating_point else fill_value
            )
        else:
            expected_pixel_value = fill_value / 255 if from_int else fill_value
        assert (td["pixels"] == expected_pixel_value).all()

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_transform_rb(self, out_keys, rbclass):
        t = ToTensorImage(in_keys=["pixels"], out_keys=out_keys)
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(t)
        td = TensorDict({"pixels": torch.randint(255, (21, 22, 3))}, [])
        rb.extend(td.expand(10))
        td = rb.sample(2)
        if out_keys is not None:
            assert out_keys[0] in td.keys()
            obs = td[out_keys[0]]
        else:
            obs = td["pixels"]
        assert obs.shape[-3] == 3
        assert obs.dtype is torch.float32

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for ToTensorImage")


class TestTimeMaxPool(TransformBase):
    @pytest.mark.parametrize("T", [2, 4])
    @pytest.mark.parametrize("seq_len", [8])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_no_env(self, T, seq_len, device):
        batch = 1
        nodes = 4
        keys = ["observation", ("nested", "key")]
        time_max_pool = TimeMaxPool(keys, T=T)

        tensor_list = []
        for _ in range(seq_len):
            tensor_list.append(torch.rand(batch, nodes).to(device))
        max_vals, _ = torch.max(torch.stack(tensor_list[-T:]), dim=0)

        for i in range(seq_len):
            env_td = TensorDict(
                {
                    "observation": tensor_list[i],
                    ("nested", "key"): tensor_list[i].clone(),
                },
                device=device,
                batch_size=[batch],
            )
            transformed_td = time_max_pool._call(env_td)

        assert (max_vals == transformed_td["observation"]).all()
        assert (max_vals == transformed_td["nested", "key"]).all()

    @pytest.mark.parametrize("T", [2, 4])
    @pytest.mark.parametrize("seq_len", [8])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_compose(self, T, seq_len, device):
        batch = 1
        nodes = 4
        keys = ["observation"]
        time_max_pool = Compose(TimeMaxPool(keys, T=T))

        tensor_list = []
        for _ in range(seq_len):
            tensor_list.append(torch.rand(batch, nodes).to(device))
        max_vals, _ = torch.max(torch.stack(tensor_list[-T:]), dim=0)

        for i in range(seq_len):
            env_td = TensorDict(
                {
                    "observation": tensor_list[i],
                },
                device=device,
                batch_size=[batch],
            )
            transformed_td = time_max_pool._call(env_td)

        assert (max_vals == transformed_td["observation"]).all()

    @pytest.mark.parametrize("out_keys", [None, ["obs2"]])
    def test_single_trans_env_check(self, out_keys):
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            TimeMaxPool(in_keys=["observation"], T=3, out_keys=out_keys),
        )
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        env = SerialEnv(
            2,
            lambda: TransformedEnv(
                ContinuousActionVecMockEnv(),
                TimeMaxPool(
                    in_keys=["observation"],
                    T=3,
                ),
            ),
        )
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        env = maybe_fork_ParallelEnv(
            2,
            lambda: TransformedEnv(
                ContinuousActionVecMockEnv(),
                TimeMaxPool(
                    in_keys=["observation"],
                    T=3,
                ),
            ),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, lambda: ContinuousActionVecMockEnv()),
            TimeMaxPool(
                in_keys=["observation"],
                T=3,
            ),
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, ContinuousActionVecMockEnv),
            TimeMaxPool(
                in_keys=["observation"],
                T=3,
            ),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.skipif(not _has_gym, reason="Test executed on gym")
    @pytest.mark.parametrize("batched_class", [ParallelEnv, SerialEnv])
    @pytest.mark.parametrize("break_when_any_done", [True, False])
    def test_timemax_batching(self, batched_class, break_when_any_done):
        env = TransformedEnv(
            batched_class(2, lambda: GymEnv(CARTPOLE_VERSIONED())),
            TimeMaxPool(
                in_keys=["observation"],
                out_keys=["observation_max"],
                T=3,
            ),
        )
        torch.manual_seed(0)
        env.set_seed(0)
        r0 = env.rollout(100, break_when_any_done=break_when_any_done)

        env = batched_class(
            2,
            lambda: TransformedEnv(
                GymEnv(CARTPOLE_VERSIONED()),
                TimeMaxPool(
                    in_keys=["observation"],
                    out_keys=["observation_max"],
                    T=3,
                ),
            ),
        )
        torch.manual_seed(0)
        env.set_seed(0)
        r1 = env.rollout(100, break_when_any_done=break_when_any_done)
        tensordict.tensordict.assert_allclose_td(r0, r1)

    @pytest.mark.skipif(not _has_gym, reason="Gym not available")
    @pytest.mark.parametrize("out_keys", [None, ["obs2"], [("some", "other")]])
    def test_transform_env(self, out_keys):
        env = TransformedEnv(
            GymEnv(PENDULUM_VERSIONED(), frame_skip=4),
            TimeMaxPool(
                in_keys=["observation"],
                out_keys=out_keys,
                T=3,
            ),
        )
        td = env.reset()
        if out_keys:
            assert td[out_keys[0]].shape[-1] == 3
        else:
            assert td["observation"].shape[-1] == 3

    def test_transform_model(self):
        key1 = "first key"
        key2 = "second key"
        keys = [key1, key2]
        dim = -2
        d = 4
        batch_size = (5,)
        extra_d = (3,) * (-dim - 1)
        device = "cpu"
        key1_tensor = torch.ones(*batch_size, d, *extra_d, device=device) * 2
        key2_tensor = torch.ones(*batch_size, d, *extra_d, device=device)
        key_tensors = [key1_tensor, key2_tensor]
        td = TensorDict(dict(zip(keys, key_tensors)), batch_size, device=device)
        t = TimeMaxPool(
            in_keys=["observation"],
            T=3,
        )

        model = nn.Sequential(t, nn.Identity())
        with pytest.raises(
            NotImplementedError, match="TimeMaxPool cannot be called independently"
        ):
            model(td)

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        key1 = "first key"
        key2 = "second key"
        keys = [key1, key2]
        dim = -2
        d = 4
        batch_size = (5,)
        extra_d = (3,) * (-dim - 1)
        device = "cpu"
        key1_tensor = torch.ones(*batch_size, d, *extra_d, device=device) * 2
        key2_tensor = torch.ones(*batch_size, d, *extra_d, device=device)
        key_tensors = [key1_tensor, key2_tensor]
        td = TensorDict(dict(zip(keys, key_tensors)), batch_size, device=device)
        t = TimeMaxPool(
            in_keys=["observation"],
            T=3,
        )
        rb = rbclass(storage=LazyTensorStorage(20))
        rb.append_transform(t)
        rb.extend(td)
        with pytest.raises(
            NotImplementedError, match="TimeMaxPool cannot be called independently"
        ):
            _ = rb.sample(10)

    @pytest.mark.parametrize("device", get_default_devices())
    def test_tmp_reset(self, device):
        key1 = "first key"
        key2 = "second key"
        keys = [key1, key2]
        key1_tensor = torch.randn(1, 1, 3, 3, device=device)
        key2_tensor = torch.randn(1, 1, 3, 3, device=device)
        key_tensors = [key1_tensor, key2_tensor]
        td = TensorDict(dict(zip(keys, key_tensors)), [1], device=device)
        t = TimeMaxPool(in_keys=key1, T=3, reset_key="_reset")

        t._call(td.clone())
        buffer = getattr(t, f"_maxpool_buffer_{key1}")

        tdc = td.clone()
        t._reset(tdc, tdc.empty())

        # assert tdc is passed_back_td
        assert (buffer != 0).any()

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for TimeMaxPool")


class TestPermuteTransform(TransformBase):
    envclass = DiscreteActionConvMockEnv

    @classmethod
    def _get_permute(cls):
        return PermuteTransform(
            (-1, -2, -3), in_keys=["pixels_orig", "pixels"], in_keys_inv=["pixels_orig"]
        )

    def test_single_trans_env_check(self):
        base_env = TestPermuteTransform.envclass()
        env = TransformedEnv(base_env, TestPermuteTransform._get_permute())
        check_env_specs(env)
        assert env.observation_spec["pixels"] == env.observation_spec["pixels_orig"]
        assert env.state_spec["pixels_orig"] == env.observation_spec["pixels_orig"]

    def test_serial_trans_env_check(self):
        env = SerialEnv(
            2,
            lambda: TransformedEnv(
                TestPermuteTransform.envclass(), TestPermuteTransform._get_permute()
            ),
        )
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        env = maybe_fork_ParallelEnv(
            2,
            lambda: TransformedEnv(
                TestPermuteTransform.envclass(), TestPermuteTransform._get_permute()
            ),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, TestPermuteTransform.envclass),
            TestPermuteTransform._get_permute(),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, TestPermuteTransform.envclass),
            TestPermuteTransform._get_permute(),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    def test_transform_compose(self, batch):
        D, W, H, C = 8, 32, 64, 3
        trans = Compose(
            PermuteTransform(
                dims=(-1, -4, -2, -3),
                in_keys=["pixels"],
            )
        )  # DxWxHxC => CxDxHxW
        td = TensorDict({"pixels": torch.randn((*batch, D, W, H, C))}, batch_size=batch)
        td = trans(td)
        assert td["pixels"].shape == torch.Size((*batch, C, D, H, W))

    def test_transform_env(self):
        base_env = TestPermuteTransform.envclass()
        env = TransformedEnv(base_env, TestPermuteTransform._get_permute())
        check_env_specs(env)
        assert env.observation_spec["pixels"] == env.observation_spec["pixels_orig"]
        assert env.state_spec["pixels_orig"] == env.observation_spec["pixels_orig"]
        assert env.state_spec["pixels_orig"] != base_env.state_spec["pixels_orig"]
        assert env.observation_spec["pixels"] != base_env.observation_spec["pixels"]

        td = env.rollout(3)
        assert td["pixels"].shape == torch.Size([3, 7, 7, 1])

        # check error
        with pytest.raises(ValueError, match="Only tailing dims with negative"):
            PermuteTransform((-1, -10))

    def test_transform_model(self):
        batch = [2]
        D, W, H, C = 8, 32, 64, 3
        trans = PermuteTransform(
            dims=(-1, -4, -2, -3),
            in_keys=["pixels"],
        )  # DxWxHxC => CxDxHxW
        td = TensorDict({"pixels": torch.randn((*batch, D, W, H, C))}, batch_size=batch)
        out_channels = 4
        model = nn.Sequential(
            trans,
            TensorDictModule(
                nn.Conv3d(C, out_channels, 3, padding=1),
                in_keys=["pixels"],
                out_keys=["pixels"],
            ),
        )
        td = model(td)
        assert td["pixels"].shape == torch.Size((*batch, out_channels, D, H, W))

    def test_transform_rb(self):
        batch = [6]
        D, W, H, C = 4, 5, 6, 3
        trans = PermuteTransform(
            dims=(-1, -4, -2, -3),
            in_keys=["pixels"],
        )  # DxWxHxC => CxDxHxW
        td = TensorDict({"pixels": torch.randn((*batch, D, W, H, C))}, batch_size=batch)
        rb = TensorDictReplayBuffer(storage=LazyTensorStorage(5), transform=trans)
        rb.extend(td)
        sample = rb.sample(2)
        assert sample["pixels"].shape == torch.Size([2, C, D, H, W])

    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    def test_transform_inverse(self, batch):
        D, W, H, C = 8, 32, 64, 3
        trans = PermuteTransform(
            dims=(-1, -4, -2, -3),
            in_keys_inv=["pixels"],
        )  # DxWxHxC => CxDxHxW
        td = TensorDict({"pixels": torch.randn((*batch, C, D, H, W))}, batch_size=batch)
        td = trans.inv(td)
        assert td["pixels"].shape == torch.Size((*batch, D, W, H, C))

    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    def test_transform_no_env(self, batch):
        D, W, H, C = 8, 32, 64, 3
        trans = PermuteTransform(
            dims=(-1, -4, -2, -3),
            in_keys=["pixels"],
        )  # DxWxHxC => CxDxHxW
        td = TensorDict({"pixels": torch.randn((*batch, D, W, H, C))}, batch_size=batch)
        td = trans(td)
        assert td["pixels"].shape == torch.Size((*batch, C, D, H, W))
