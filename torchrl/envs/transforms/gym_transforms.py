# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Gym-specific transforms."""
import warnings

import torch
import torchrl.objectives.common
from tensordict import TensorDictBase
from tensordict.utils import expand_as_right, NestedKey
from torchrl.data.tensor_specs import UnboundedDiscreteTensorSpec

from torchrl.envs.transforms.transforms import FORWARD_NOT_IMPLEMENTED, Transform


class EndOfLifeTransform(Transform):
    """Registers the end-of-life signal from a Gym env with a `lives` method.

    Proposed by DeepMind for the DQN and co. It helps value estimation.

    Args:
        eol_key (NestedKey, optional): the key where the end-of-life signal should
            be written. Defaults to ``"end-of-life"``.
        done_key (NestedKey, optional): a "done" key in the parent env done_spec,
            where the done value can be retrieved. This key must be unique and its
            shape must match the shape of the end-of-life entry. Defaults to ``"done"``.
        eol_attribute (str, optional): the location of the "lives" in the gym env.
            Defaults to ``"unwrapped.ale.lives"``. Supported attribute types are
            integer/array-like objects or callables that return these values.

    .. note::
        This transform should be used with gym envs that have a ``env.unwrapped.ale.lives``.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> from torchrl.envs.transforms.transforms import TransformedEnv
        >>> env = GymEnv("ALE/Breakout-v5")
        >>> env.rollout(100)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([100, 4]), device=cpu, dtype=torch.int64, is_shared=False),
                done: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        pixels: Tensor(shape=torch.Size([100, 210, 160, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
                        reward: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([100]),
                    device=cpu,
                    is_shared=False),
                pixels: Tensor(shape=torch.Size([100, 210, 160, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
                terminated: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([100]),
            device=cpu,
            is_shared=False)
        >>> eol_transform = EndOfLifeTransform()
        >>> env = TransformedEnv(env, eol_transform)
        >>> env.rollout(100)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([100, 4]), device=cpu, dtype=torch.int64, is_shared=False),
                done: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                eol: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                lives: Tensor(shape=torch.Size([100]), device=cpu, dtype=torch.int64, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        end-of-life: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        lives: Tensor(shape=torch.Size([100]), device=cpu, dtype=torch.int64, is_shared=False),
                        pixels: Tensor(shape=torch.Size([100, 210, 160, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
                        reward: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([100]),
                    device=cpu,
                    is_shared=False),
                pixels: Tensor(shape=torch.Size([100, 210, 160, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
                terminated: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([100]),
            device=cpu,
            is_shared=False)

    The typical usage of this transform is to replace the "done" state by "end-of-life"
    within the loss module. The end-of-life signal isn't registered within the ``done_spec``
    because it should not instruct the env to reset.

    Examples:
        >>> from torchrl.objectives import DQNLoss
        >>> module = torch.nn.Identity() # used as a placeholder
        >>> loss = DQNLoss(module, action_space="categorical")
        >>> loss.set_keys(done="end-of-life", terminated="end-of-life")
        >>> # equivalently
        >>> eol_transform.register_keys(loss)
    """

    NO_PARENT_ERR = "The {} transform is being executed without a parent env. This is currently not supported."

    def __init__(
        self,
        eol_key: NestedKey = "end-of-life",
        lives_key: NestedKey = "lives",
        done_key: NestedKey = "done",
        eol_attribute="unwrapped.ale.lives",
    ):
        super().__init__(in_keys=[done_key], out_keys=[eol_key, lives_key])
        self.eol_key = eol_key
        self.lives_key = lives_key
        self.done_key = done_key
        self.eol_attribute = eol_attribute.split(".")

    def _get_lives(self):
        from torchrl.envs.libs.gym import GymWrapper

        base_env = self.parent.base_env
        if not isinstance(base_env, GymWrapper):
            warnings.warn(
                f"The base_env is not a gym env. Compatibility of {type(self)} is not guaranteed with "
                f"environment types that do not inherit from GymWrapper.",
                category=UserWarning,
            )
        # getattr falls back on _env by default
        lives = getattr(base_env, self.eol_attribute[0])
        for att in self.eol_attribute[1:]:
            if isinstance(lives, list):
                # For SerialEnv (and who knows Parallel one day)
                lives = [getattr(_lives, att) for _lives in lives]
            else:
                lives = getattr(lives, att)
        if callable(lives):
            lives = lives()
        elif isinstance(lives, list) and all(callable(_lives) for _lives in lives):
            lives = torch.tensor([_lives() for _lives in lives])
        return lives

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def _step(self, tensordict, next_tensordict):
        parent = self.parent
        if parent is None:
            raise RuntimeError(self.NO_PARENT_ERR.format(type(self)))

        lives = self._get_lives()
        end_of_life = torch.tensor(
            tensordict.get(self.lives_key) > lives, device=self.parent.device
        )
        try:
            done = next_tensordict.get(self.done_key)
        except KeyError:
            raise KeyError(
                f"The done value pointed by {self.done_key} cannot be found in tensordict with keys {tensordict.keys(True, True)}. "
                f"Make sure to pass the appropriate done_key to the {type(self)} transform."
            )
        end_of_life = expand_as_right(end_of_life, done) | done
        next_tensordict.set(self.eol_key, end_of_life)
        next_tensordict.set(self.lives_key, lives)
        return next_tensordict

    def _reset(self, tensordict, tensordict_reset):
        parent = self.parent
        if parent is None:
            raise RuntimeError(self.NO_PARENT_ERR.format(type(self)))
        lives = self._get_lives()
        end_of_life = False
        tensordict_reset.set(
            self.eol_key,
            torch.tensor(end_of_life).expand(
                parent.full_done_spec[self.done_key].shape
            ),
        )
        tensordict_reset.set(self.lives_key, lives)
        return tensordict_reset

    def transform_observation_spec(self, observation_spec):
        full_done_spec = self.parent.output_spec["full_done_spec"]
        observation_spec[self.eol_key] = full_done_spec[self.done_key].clone()
        observation_spec[self.lives_key] = UnboundedDiscreteTensorSpec(
            self.parent.batch_size,
            device=self.parent.device,
            dtype=torch.int64,
        )
        return observation_spec

    def register_keys(self, loss_or_advantage: "torchrl.objectives.common.LossModule"):
        """Registers the end-of-life key at appropriate places within the loss.

        Args:
            loss_or_advantage (torchrl.objectives.LossModule or torchrl.objectives.value.ValueEstimatorBase): a module to instruct what the end-of-life key is.

        """
        loss_or_advantage.set_keys(done=self.eol_key, terminated=self.eol_key)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise RuntimeError(FORWARD_NOT_IMPLEMENTED.format(type(self)))
