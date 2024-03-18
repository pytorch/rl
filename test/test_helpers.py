# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import dataclasses
import pathlib
import sys

from time import sleep

import pytest
import torch
from _utils_internal import generate_seeds, get_default_devices
from torchrl._utils import timeit

try:
    from hydra import compose, initialize
    from hydra.core.config_store import ConfigStore

    _has_hydra = True
except ImportError:
    _has_hydra = False
from mocking_classes import (
    ContinuousActionConvMockEnvNumpy,
    ContinuousActionVecMockEnv,
    DiscreteActionConvMockEnvNumpy,
    DiscreteActionVecMockEnv,
    MockSerialEnv,
)
from packaging import version
from torchrl.data import BoundedTensorSpec, CompositeSpec
from torchrl.envs.libs.gym import _has_gym
from torchrl.envs.transforms import ObservationNorm
from torchrl.envs.transforms.transforms import (
    _has_tv,
    FlattenObservation,
    TransformedEnv,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules.tensordict_module.common import _has_functorch
from torchrl.trainers.helpers import transformed_env_constructor
from torchrl.trainers.helpers.envs import (
    EnvConfig,
    initialize_observation_norm_transforms,
    retrieve_observation_norms_state_dict,
)
from torchrl.trainers.helpers.models import (
    DiscreteModelConfig,
    DreamerConfig,
    make_dqn_actor,
    make_dreamer,
    make_redq_model,
    REDQModelConfig,
)

TORCH_VERSION = version.parse(torch.__version__)
if TORCH_VERSION < version.parse("1.12.0"):
    UNSQUEEZE_SINGLETON = True
else:
    UNSQUEEZE_SINGLETON = False


## these tests aren't truly unitary but setting up a fake env for the
# purpose of building a model with args is a lot of unstable scaffoldings
# with unclear benefits


@pytest.fixture
def dreamer_constructor_fixture():

    # we hack the env constructor
    import sys

    sys.path.append(
        str(pathlib.Path(__file__).parent.parent / "sota-implementations" / "dreamer")
    )

    from dreamer_utils import transformed_env_constructor

    yield transformed_env_constructor
    sys.path.pop()


@pytest.mark.skipif(not _has_gym, reason="No gym library found")
@pytest.mark.skipif(not _has_tv, reason="No torchvision library found")
@pytest.mark.skipif(not _has_hydra, reason="No hydra library found")
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("noisy", [(), ("noisy=True",)])
@pytest.mark.parametrize("distributional", [(), ("distributional=True",)])
@pytest.mark.parametrize("from_pixels", [(), ("from_pixels=True", "catframes=4")])
@pytest.mark.parametrize(
    "categorical_action_encoding",
    [("categorical_action_encoding=True",), ("categorical_action_encoding=False",)],
)
def test_dqn_maker(
    device, noisy, distributional, from_pixels, categorical_action_encoding
):
    flags = list(noisy + distributional + from_pixels + categorical_action_encoding) + [
        "env_name=CartPole-v1"
    ]

    config_fields = [
        (config_field.name, config_field.type, config_field)
        for config_cls in (
            EnvConfig,
            DiscreteModelConfig,
        )
        for config_field in dataclasses.fields(config_cls)
    ]

    Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    with initialize(version_base="1.1", config_path=None):
        cfg = compose(config_name="config", overrides=flags)

        env_maker = (
            DiscreteActionConvMockEnvNumpy if from_pixels else DiscreteActionVecMockEnv
        )
        env_maker = transformed_env_constructor(
            cfg,
            use_env_creator=False,
            custom_env_maker=env_maker,
            stats={"loc": 0.0, "scale": 1.0},
        )
        proof_environment = env_maker(
            categorical_action_encoding=cfg.categorical_action_encoding,
        )

        actor = make_dqn_actor(proof_environment, cfg, device)
        td = proof_environment.reset().to(device)
        if UNSQUEEZE_SINGLETON and not td.ndimension():
            # Linear and conv used to break for non-batched data
            actor(td.unsqueeze(0))
        else:
            actor(td)

        expected_keys = [
            "done",
            "terminated",
            "action",
            "action_value",
            "step_count",
            "is_init",
        ]
        if from_pixels:
            expected_keys += [
                "pixels",
                "pixels_orig",
            ]
        else:
            expected_keys += ["observation_orig", "observation_vector"]

        if not distributional:
            expected_keys += ["chosen_action_value"]
        try:
            assert set(td.keys()) == set(expected_keys)
        except AssertionError:
            proof_environment.close()
            raise
        proof_environment.close()


@pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
@pytest.mark.skipif(not _has_hydra, reason="No hydra library found")
@pytest.mark.skipif(not _has_gym, reason="No gym library found")
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("from_pixels", [(), ("from_pixels=True", "catframes=4")])
@pytest.mark.parametrize("gsde", [(), ("gSDE=True",)])
@pytest.mark.parametrize("exploration", [ExplorationType.MODE, ExplorationType.RANDOM])
def test_redq_make(device, from_pixels, gsde, exploration):
    if not gsde and exploration != ExplorationType.RANDOM:
        pytest.skip("no need to test this setting")
    flags = list(from_pixels + gsde)
    if gsde and from_pixels:
        pytest.skip("gsde and from_pixels are incompatible")

    config_fields = [
        (config_field.name, config_field.type, config_field)
        for config_cls in (
            EnvConfig,
            REDQModelConfig,
        )
        for config_field in dataclasses.fields(config_cls)
    ]

    Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    with initialize(version_base="1.1", config_path=None):
        cfg = compose(config_name="config", overrides=flags)

        env_maker = (
            ContinuousActionConvMockEnvNumpy
            if from_pixels
            else ContinuousActionVecMockEnv
        )
        env_maker = transformed_env_constructor(
            cfg,
            use_env_creator=False,
            custom_env_maker=env_maker,
            stats={"loc": 0.0, "scale": 1.0},
        )
        proof_environment = env_maker()

        model = make_redq_model(
            proof_environment,
            device=device,
            cfg=cfg,
        )
        actor, qvalue = model
        td = proof_environment.reset().to(device)
        with set_exploration_type(exploration):
            actor(td)
        expected_keys = [
            "done",
            "terminated",
            "action",
            "sample_log_prob",
            "loc",
            "scale",
            "step_count",
            "is_init",
        ]
        if len(gsde):
            expected_keys += ["_eps_gSDE"]
        if from_pixels:
            expected_keys += [
                "hidden",
                "pixels",
                "pixels_orig",
            ]
        else:
            expected_keys += ["observation_vector", "observation_orig"]

        try:
            assert set(td.keys()) == set(expected_keys)
        except AssertionError:
            proof_environment.close()
            raise

        if cfg.gSDE:
            tsf_loc = actor.module[0].module[-1].module.transform(td.get("loc"))
            if exploration == ExplorationType.RANDOM:
                with pytest.raises(AssertionError):
                    torch.testing.assert_close(td.get("action"), tsf_loc)
            else:
                torch.testing.assert_close(td.get("action"), tsf_loc)

        qvalue(td)
        expected_keys = [
            "done",
            "terminated",
            "action",
            "sample_log_prob",
            "state_action_value",
            "loc",
            "scale",
            "step_count",
            "is_init",
        ]
        if len(gsde):
            expected_keys += ["_eps_gSDE"]
        if from_pixels:
            expected_keys += [
                "hidden",
                "pixels",
                "pixels_orig",
            ]
        else:
            expected_keys += ["observation_vector", "observation_orig"]
        try:
            assert set(td.keys()) == set(expected_keys)
        except AssertionError:
            proof_environment.close()
            raise
        proof_environment.close()
        del proof_environment


@pytest.mark.skipif(not _has_hydra, reason="No hydra library found")
@pytest.mark.skipif(not _has_gym, reason="No gym library found")
@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("1.11.0"),
    reason="""Dreamer works with batches of null to 2 dimensions. Torch < 1.11
requires one-dimensional batches (for RNN and Conv nets for instance). If you'd like
to see torch < 1.11 supported for dreamer, please submit an issue.""",
)
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("tanh_loc", [(), ("tanh_loc=True",)])
@pytest.mark.parametrize("exploration", [ExplorationType.MODE, ExplorationType.RANDOM])
def test_dreamer_make(device, tanh_loc, exploration, dreamer_constructor_fixture):
    transformed_env_constructor = dreamer_constructor_fixture
    flags = ["from_pixels=True", "catframes=1"]

    config_fields = [
        (config_field.name, config_field.type, config_field)
        for config_cls in (
            EnvConfig,
            DreamerConfig,
        )
        for config_field in dataclasses.fields(config_cls)
    ]

    Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    with initialize(version_base="1.1", config_path=None):
        cfg = compose(config_name="config", overrides=flags)
        env_maker = ContinuousActionConvMockEnvNumpy
        env_maker = transformed_env_constructor(
            cfg,
            use_env_creator=False,
            custom_env_maker=env_maker,
            stats={"loc": 0.0, "scale": 1.0},
        )
        proof_environment = env_maker().to(device)
        model = make_dreamer(
            proof_environment=proof_environment,
            device=device,
            cfg=cfg,
        )
        world_model, model_based_env, actor_model, value_model, policy = model
        out = world_model(proof_environment.rollout(3))
        expected_keys = {
            "action",
            "belief",
            "done",
            "terminated",
            ("next", "done"),
            ("next", "terminated"),
            ("next", "reward"),
            ("next", "belief"),
            ("next", "encoded_latents"),
            ("next", "pixels"),
            ("next", "pixels_orig"),
            ("next", "posterior_mean"),
            ("next", "posterior_std"),
            ("next", "prior_mean"),
            ("next", "prior_std"),
            ("next", "state"),
            "pixels",
            "pixels_orig",
            "state",
            ("next", "reco_pixels"),
            "next",
        }
        assert set(out.keys(True)) == expected_keys

        simulated_data = model_based_env.rollout(3)
        expected_keys = {
            "action",
            "belief",
            "done",
            "terminated",
            ("next", "done"),
            ("next", "terminated"),
            ("next", "reward"),
            ("next", "belief"),
            ("next", "state"),
            ("next", "pixels"),
            ("next", "pixels_orig"),
            "pixels_orig",
            "pixels",
            "state",
            "next",
        }
        assert expected_keys == set(simulated_data.keys(True))

        simulated_action = actor_model(model_based_env.reset())
        real_action = actor_model(proof_environment.reset())
        simulated_policy_action = policy(model_based_env.reset())
        real_policy_action = policy(proof_environment.reset())
        assert "action" in simulated_action.keys()
        assert "action" in real_action.keys()
        assert "action" in simulated_policy_action.keys()
        assert "action" in real_policy_action.keys()

        value_td = value_model(proof_environment.reset())
        assert "state_value" in value_td.keys()


@pytest.mark.parametrize("initial_seed", range(5))
def test_seed_generator(initial_seed):
    num_seeds = 100

    # Check unique seed generation
    if initial_seed == 0:
        with pytest.raises(ValueError):
            generate_seeds(initial_seed - 1, num_seeds)
        return
    else:
        seeds0 = generate_seeds(initial_seed - 1, num_seeds)
    seeds1 = generate_seeds(initial_seed, num_seeds)
    assert len(seeds1) == num_seeds
    assert len(seeds1) == len(set(seeds1))
    assert len(set(seeds0).intersection(set(seeds1))) == 0

    # Check deterministic seed generation
    seeds0 = generate_seeds(initial_seed, num_seeds)
    seeds1 = generate_seeds(initial_seed, num_seeds)
    assert seeds0 == seeds1


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="val1[0]-w1 consistently ~0.015 (> 0.01) in CI pipeline on Windows machine",
)
def test_timeit():
    n1 = 500
    w1 = 1e-4
    n2 = 200
    w2 = 1e-4
    w3 = 1e-4
    # warmup
    for _ in range(10):
        sleep(w1)
    for _ in range(n1):
        with timeit("event1"):
            sleep(w1)
        sleep(w3)
    for _ in range(n2):
        with timeit("event2"):
            sleep(w2)
    val1 = timeit._REG["event1"]
    val2 = timeit._REG["event2"]
    assert abs(val1[0] - w1) < 1e-2
    assert abs(val1[1] - n1 * w1) < 1
    assert val1[2] == n1
    assert abs(val2[0] - w2) < 1e-2
    assert abs(val2[1] - n2 * w2) < 1
    assert val2[2] == n2


@pytest.mark.skipif(not _has_hydra, reason="No hydra library found")
@pytest.mark.parametrize("from_pixels", [(), ("from_pixels=True", "catframes=4")])
def test_transformed_env_constructor_with_state_dict(from_pixels):
    config_fields = [
        (config_field.name, config_field.type, config_field)
        for config_cls in (
            EnvConfig,
            DreamerConfig,
        )
        for config_field in dataclasses.fields(config_cls)
    ]
    flags = list(from_pixels)

    Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    with initialize(version_base="1.1", config_path=None):
        cfg = compose(config_name="config", overrides=flags)
        env_maker = (
            ContinuousActionConvMockEnvNumpy
            if from_pixels
            else ContinuousActionVecMockEnv
        )
        t_env = transformed_env_constructor(
            cfg,
            use_env_creator=False,
            custom_env_maker=env_maker,
        )()
        for t in t_env.transform:
            if isinstance(t, ObservationNorm):
                t.init_stats(4)
        idx, state_dict = retrieve_observation_norms_state_dict(t_env)[0]

        obs_transform = transformed_env_constructor(
            cfg,
            use_env_creator=False,
            custom_env_maker=env_maker,
            obs_norm_state_dict=state_dict,
        )().transform[idx]
        torch.testing.assert_close(obs_transform.state_dict(), state_dict)


@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("keys", [None, ["observation", "observation_orig"]])
@pytest.mark.parametrize("composed", [True, False])
@pytest.mark.parametrize("initialized", [True, False])
def test_initialize_stats_from_observation_norms(device, keys, composed, initialized):
    obs_spec, stat_key = None, None
    if keys:
        obs_spec = CompositeSpec(
            **{
                key: BoundedTensorSpec(high=1, low=1, shape=torch.Size([1]))
                for key in keys
            }
        )
        stat_key = keys[0]
        env = ContinuousActionVecMockEnv(
            device=device,
            observation_spec=obs_spec,
            action_spec=BoundedTensorSpec(low=1, high=2, shape=torch.Size((1,))),
        )
        env.out_key = "observation"
    else:
        env = MockSerialEnv(device=device)
        env.set_seed(1)

    t_env = TransformedEnv(env)
    stats = {"loc": None, "scale": None}
    if initialized:
        stats = {"loc": 0.0, "scale": 1.0}
    t_env.transform = ObservationNorm(
        in_keys=["observation"], standard_normal=True, **stats
    )
    if composed:
        t_env.append_transform(
            ObservationNorm(in_keys=["observation"], standard_normal=True, **stats)
        )
    if not initialized:
        with pytest.raises(
            ValueError, match="Attempted to use an uninitialized parameter"
        ):
            t_env.transform.state_dict()
        return
    pre_init_state_dict = t_env.transform.state_dict()
    initialize_observation_norm_transforms(
        proof_environment=t_env, num_iter=100, key=stat_key
    )
    post_init_state_dict = t_env.transform.state_dict()
    expected_dict_size = 4 if composed else 2
    expected_dict_size = expected_dict_size if not initialized else 0

    assert len(post_init_state_dict) == len(pre_init_state_dict) + expected_dict_size


@pytest.mark.parametrize("device", get_default_devices())
def test_initialize_stats_from_non_obs_transform(device):
    env = MockSerialEnv(device=device)
    env.set_seed(1)

    t_env = TransformedEnv(env)
    t_env.transform = FlattenObservation(
        first_dim=0, last_dim=-3, allow_positive_dim=True
    )
    pre_init_state_dict = t_env.transform.state_dict()
    initialize_observation_norm_transforms(proof_environment=t_env, num_iter=100)
    post_init_state_dict = t_env.transform.state_dict()
    assert len(post_init_state_dict) == len(pre_init_state_dict)


def test_initialize_obs_transform_stats_raise_exception():
    env = ContinuousActionVecMockEnv()
    t_env = TransformedEnv(env)
    t_env.transform = ObservationNorm(in_keys=["observation"])
    with pytest.raises(
        RuntimeError, match="More than one key exists in the observation_specs"
    ):
        initialize_observation_norm_transforms(proof_environment=t_env, num_iter=100)


@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("composed", [True, False])
def test_retrieve_observation_norms_state_dict(device, composed):
    env = MockSerialEnv(device=device)
    env.set_seed(1)

    t_env = TransformedEnv(env)
    t_env.transform = ObservationNorm(
        standard_normal=True, loc=0.5, scale=0.2, in_keys=["observation"]
    )
    if composed:
        t_env.append_transform(
            ObservationNorm(
                standard_normal=True, loc=1.0, scale=0.3, in_keys=["observation"]
            )
        )
    initialize_observation_norm_transforms(proof_environment=t_env, num_iter=100)
    state_dicts = retrieve_observation_norms_state_dict(t_env)
    expected_state_count = 2 if composed else 1
    expected_idx = [0, 1] if composed else [0]

    assert len(state_dicts) == expected_state_count
    for idx, state_dict in enumerate(state_dicts):
        assert len(state_dict[1]) == 3
        assert state_dict[0] == expected_idx[idx]


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
