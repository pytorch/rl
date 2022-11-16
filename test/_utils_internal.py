# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from functools import wraps

# Get relative file path
# this returns relative path from current file.
import pytest
import torch.cuda
from tensordict.tensordict import TensorDictBase
from torchrl._utils import seed_generator
from torchrl.envs import EnvBase
from torchrl.data import CompositeSpec


# Specified for test_utils.py
__version__ = "0.3"


def get_relative_path(curr_file, *path_components):
    return os.path.join(os.path.dirname(curr_file), *path_components)


def get_available_devices():
    devices = [torch.device("cpu")]
    n_cuda = torch.cuda.device_count()
    if n_cuda > 0:
        for i in range(n_cuda):
            devices += [torch.device(f"cuda:{i}")]
    return devices


def generate_seeds(seed, repeat):
    seeds = [seed]
    for _ in range(repeat - 1):
        seed = seed_generator(seed)
        seeds.append(seed)
    return seeds


def _test_fake_tensordict(env: EnvBase):
    fake_tensordict = env.fake_tensordict().flatten_keys(".")
    real_tensordict = env.rollout(3).flatten_keys(".")

    keys1 = set(fake_tensordict.keys())
    keys2 = set(real_tensordict.keys())
    assert keys1 == keys2
    fake_tensordict = fake_tensordict.unsqueeze(real_tensordict.batch_dims - 1)
    fake_tensordict = fake_tensordict.expand(*real_tensordict.shape)
    fake_tensordict = fake_tensordict.to_tensordict()
    fake_tensordict.zero_()
    real_tensordict.zero_()
    assert (fake_tensordict == real_tensordict).all()
    for key in keys2:
        assert fake_tensordict[key].shape == real_tensordict[key].shape

    # test dtypes
    for key, value in real_tensordict.unflatten_keys(".").items():
        _check_dtype(key, value, env.observation_spec, env.input_spec)


def _check_dtype(key, value, obs_spec, input_spec):
    if key in {"reward", "done"}:
        return
    elif key == "next":
        for _key, _value in value.items():
            _check_dtype(_key, _value, obs_spec, input_spec)
        return
    elif key in input_spec.keys(yield_nesting_keys=True):
        assert input_spec[key].is_in(value)
        return
    elif key in obs_spec.keys(yield_nesting_keys=True):
        assert obs_spec[key].is_in(value)
        return
    else:
        raise KeyError(key)
    #
    # if isinstance(value, TensorDictBase) and key == "next":
    #     for _key, _value in value.items():
    #         _check_dtype(_key, _value, obs_spec, input_spec=input_spec)
    # elif isinstance(value, TensorDictBase) and isinstance(obs_spec, CompositeSpec) and key in obs_spec.keys():
    #     for _key, _value in value.items():
    #         _check_dtype(_key, _value, obs_spec=obs_spec[key], input_spec=None)
    # elif isinstance(value, TensorDictBase) and isinstance(input_spec, CompositeSpec) and key in input_spec.keys():
    #     for _key, _value in value.items():
    #         _check_dtype(_key, _value, obs_spec=None, input_spec=input_spec[key])
    # else:
    #     if isinstance(obs_spec, CompositeSpec) and key in obs_spec.keys():
    #         assert (
    #             obs_spec[key].dtype is value.dtype
    #         ), f"{obs_spec[key].dtype} vs {value.dtype} for {key}"
    #         assert obs_spec[key].is_in(value)
    #     elif isinstance(input_spec, CompositeSpec) and key in input_spec.keys():
    #         assert (
    #             input_spec[key].dtype is value.dtype
    #         ), f"{input_spec[key].dtype} vs {value.dtype} for {key}"
    #         assert input_spec[key].is_in(value)
    #     else:
    #         assert key in {"done", "reward"}, (key, value, obs_spec, input_spec)


# Decorator to retry upon certain Exceptions.
def retry(ExceptionToCheck, tries=3, delay=3, skip_after_retries=False):
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except ExceptionToCheck as e:
                    msg = "%s, Retrying in %d seconds..." % (str(e), mdelay)
                    print(msg)
                    time.sleep(mdelay)
                    mtries -= 1
            try:
                return f(*args, **kwargs)
            except ExceptionToCheck as e:
                if skip_after_retries:
                    raise pytest.skip(
                        f"Skipping after {tries} consecutive {str(e)}"
                    ) from e
                else:
                    raise e

        return f_retry  # true decorator

    return deco_retry


@pytest.fixture
def dtype_fixture():
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    yield dtype
    torch.set_default_dtype(dtype)
