# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copied from gym > 0.19 release

# this file should only be accessed when gym is installed

import collections
import copy
from collections.abc import MutableMapping

import numpy as np

IMPORT_ERROR = None
try:
    # rule of thumbs: gym precedes
    from gym import ObservationWrapper, spaces
except ImportError as err:
    IMPORT_ERROR = err
    try:
        from gymnasium import ObservationWrapper, spaces
    except ImportError as err2:
        raise err from err2

STATE_KEY = "observation"


class GymPixelObservationWrapper(ObservationWrapper):
    """Augment observations by pixel values.

    Args:
        env: The environment to wrap.
        pixels_only: If ``True`` (default), the original observation returned
            by the wrapped environment will be discarded, and a dictionary
            observation will only include pixels. If ``False``, the
            observation dictionary will contain both the original
            observations and the pixel observations.
        render_kwargs: Optional :obj:`dict` containing keyword arguments passed
            to the :obj:`self.render` method.
        pixel_keys: Optional custom string specifying the pixel
            observation's key in the :obj:`OrderedDict` of observations.
            Defaults to 'pixels'.

    Raises:
        ValueError: If :obj:`env`'s observation spec is not compatible with the
            wrapper. Supported formats are a single array, or a dict of
            arrays.
        ValueError: If :obj:`env`'s observation already contains any of the
            specified :obj:`pixel_keys`.
    """

    def __init__(
        self, env, pixels_only=True, render_kwargs=None, pixel_keys=("pixels",)
    ):
        env.reset()
        super().__init__(env)

        if render_kwargs is None:
            render_kwargs = {}

        for key in pixel_keys:
            render_kwargs.setdefault(key, {})

            render_mode = render_kwargs[key].pop("mode", "rgb_array")
            if render_mode != "rgb_array":
                raise ValueError(
                    f"Expected render_mode to be 'rgb_array', git {render_mode}"
                )
            render_kwargs[key]["mode"] = "rgb_array"

        wrapped_observation_space = env.observation_space

        if isinstance(wrapped_observation_space, spaces.Box):
            self._observation_is_dict = False
            invalid_keys = {STATE_KEY}
        elif isinstance(wrapped_observation_space, (spaces.Dict, MutableMapping)):
            self._observation_is_dict = True
            invalid_keys = set(wrapped_observation_space.spaces.keys())
        else:
            raise ValueError("Unsupported observation space structure.")

        if not pixels_only:
            # Make sure that now keys in the `pixel_keys` overlap with
            # `observation_keys`
            overlapping_keys = set(pixel_keys) & set(invalid_keys)
            if overlapping_keys:
                raise ValueError(
                    f"Duplicate or reserved pixel keys {overlapping_keys!r}."
                )

        if pixels_only:
            self.observation_space = spaces.Dict()
        elif self._observation_is_dict:
            self.observation_space = copy.deepcopy(wrapped_observation_space)
        else:
            self.observation_space = spaces.Dict()
            self.observation_space.spaces[STATE_KEY] = wrapped_observation_space

        # Extend observation space with pixels.

        pixels_spaces = {}
        for pixel_key in pixel_keys:
            pixels = self.env.render(**render_kwargs[pixel_key])

            if np.issubdtype(pixels.dtype, np.integer):
                low, high = (0, 255)
            elif np.issubdtype(pixels.dtype, np.float):
                low, high = (-float("inf"), float("inf"))
            else:
                raise TypeError(pixels.dtype)

            pixels_space = spaces.Box(
                shape=pixels.shape, low=low, high=high, dtype=pixels.dtype
            )
            pixels_spaces[pixel_key] = pixels_space

        self.observation_space.spaces.update(pixels_spaces)

        self._env = env
        self._pixels_only = pixels_only
        self._render_kwargs = render_kwargs
        self._pixel_keys = pixel_keys

    def observation(self, observation):
        pixel_observation = self._add_pixel_observation(observation)
        return pixel_observation

    def _add_pixel_observation(self, wrapped_observation):
        if self._pixels_only:
            observation = collections.OrderedDict()
        elif self._observation_is_dict:
            observation = type(wrapped_observation)(wrapped_observation)
        else:
            observation = collections.OrderedDict()
            observation[STATE_KEY] = wrapped_observation

        pixel_observations = {
            pixel_key: self.env.render(**self._render_kwargs[pixel_key])
            for pixel_key in self._pixel_keys
        }

        observation.update(pixel_observations)

        return observation
