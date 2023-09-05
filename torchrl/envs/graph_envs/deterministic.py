import abc
from typing import Optional, Callable

import torch

from torchrl.data import exclude_private
from torchrl.data.tensordict.tensordict import TensorDictBase
from torchrl.envs import EnvBase


class DeterministicGraphEnv(EnvBase):
    """A graph environment where state-action-state transitions are deterministic.

    As other graph envs, the step function of these classes will not write observations
    as a set of `"next_"` keys but it'll write the outcomes in a `"children.action"`
    sub-tensordict. Then, navigating the graph can be simply achieved via tensordict
    inexing.

    """

    @property
    def is_done(self):
        raise RuntimeError(
            "is_done is not implemented for stateless environments. "
            "This information can only be retrieved from the collected data."
        )

    def reset(
        self,
        tensordict: Optional[TensorDictBase] = None,
        execute_step: bool = True,
        **kwargs,
    ) -> TensorDictBase:
        """Resets the environment.

        As for step and _step, only the private method :obj:`_reset` should be overwritten by EnvBase subclasses.

        Args:
            tensordict (TensorDictBase, optional): tensordict to be used to contain the resulting new observation.
                In some cases, this input can also be used to pass argument to the reset function.
            execute_step (bool, optional): :obj:`"True"` is the only accepted behaviour for this class.
                The argument is kept for coherence with the parent class.
            kwargs (optional): other arguments to be passed to the native
                reset function.

        Returns:
            a tensordict (or the input tensordict, if any), modified in place with the resulting observations.

        """
        tensordict_reset = self._reset(tensordict, **kwargs)
        if tensordict_reset.device != self.device:
            tensordict_reset = tensordict_reset.to(self.device)
        if tensordict_reset is tensordict:
            raise RuntimeError(
                "EnvBase._reset should return outplace changes to the input "
                "tensordict. Consider emptying the TensorDict first (e.g. tensordict.empty() or "
                "tensordict.select()) inside _reset before writing new tensors onto this new instance."
            )
        if not isinstance(tensordict_reset, TensorDictBase):
            raise RuntimeError(
                f"env._reset returned an object of type {type(tensordict_reset)} but a TensorDict was expected."
            )

        is_done = tensordict_reset.get(
            "prev_done",
            torch.zeros(self.batch_size, dtype=torch.bool, device=self.device),
        )
        if is_done:
            raise RuntimeError(
                f"Env {self} was done after reset. This is (currently) not allowed."
            )
        if not execute_step:
            raise NotImplementedError(
                "`execute_step=False` is not implemented for objects of type"
                f"{self.__class__.__name__}."
            )
        if tensordict is not None:
            tensordict.update(tensordict_reset)
        else:
            tensordict = tensordict_reset
        return tensordict

    def step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Makes a step in the environment.

        Args:
            tensordict (TensorDictBase): Tensordict containing the action to be taken.

        Returns:
            the input tensordict, modified in place with the :doc:`"_children"` key-value pair
            updated.

        """
        # sanity check
        self._assert_tensordict_shape(tensordict)

        tensordict.is_locked = True  # make sure _step does not modify the tensordict
        tensordict_out = self._step(tensordict)
        tensordict.is_locked = False

        if tensordict_out is tensordict:
            raise RuntimeError(
                "EnvBase._step should return outplace changes to the input "
                "tensordict. Consider emptying the TensorDict first (e.g. tensordict.empty() or "
                "tensordict.select()) inside _step before writing new tensors onto this new instance."
            )
        # get the child node
        action_hash = self._hash_action(tensordict.get("action"))
        child = tensordict_out["_children", action_hash]
        tensordict_out.update(
            {
                "reward": child["prev_reward"],
                "done": child["prev_done"],
            }
        )
        if self.run_type_checks:
            # TODO: test that the parent tensordict has keys "done" and "reward" too
            for key in self._select_observation_keys(child):
                obs = child.get(key)
                self.observation_spec.type_check(obs, key)

            if child._get_meta("prev_reward").dtype is not self.reward_spec.dtype:
                raise TypeError(
                    f"expected reward.dtype to be {self.reward_spec.dtype} "
                    f"but got {child.get('prev_reward').dtype}"
                )

            if child._get_meta("prev_done").dtype is not torch.bool:
                raise TypeError(
                    f"expected done.dtype to be torch.bool but got {child.get('prev_done').dtype}"
                )
        tensordict.update(tensordict_out, inplace=self._inplace_update)

        return tensordict

    @abc.abstractmethod
    def _hash_action(self, action: torch.Tensor) -> str:
        """A hash function for the action.

        Returns a string that deterministically points to a specific action.
        This is used to build graphs with action keys that are specific to an action,
        though not being (necessarily) the str representation of the action.

        Examples:
            >>> class DiscreteEnv(DeterministicGraphEnv):
            ...     def _hash_action(self, action: Union[torch.Tensor, int]):
            ...         hash = str(action)
            ...         return hash

        For continuous, multi-dimensional tensors, have a look at:
        https://discuss.pytorch.org/t/defining-hash-function-for-multi-dimensional-tensor/107531/2

        """
        raise NotImplementedError

    @abc.abstractmethod
    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        """Runs a step in the environment and updates the tensordict accodringly.

        The result of the step with action provided will be written in:
        :obj:`tensordict["_children", self._hash_action(action)]`.

        A crucial difference between this method and the EnvBase._step method is
        that here it is expected that the :obj:`"prev_reward"` and :obj:`"prev_done"`
        keys will be written in the child tensordict. The parent :obj:`step` method
        will take care of copying these values in the parent tensordict.
        """
        raise NotImplementedError

    def rollout(
        self,
        max_steps: int,
        policy: Optional[Callable[[TensorDictBase], TensorDictBase]] = None,
        callback: Optional[Callable[[TensorDictBase], TensorDictBase]] = None,
        auto_reset: bool = True,
        auto_cast_to_device: bool = False,
        break_when_any_done: bool = True,
        return_contiguous: bool = True,
        tensordict: Optional[TensorDictBase] = None,
        exclude_private_keys: bool = True,
        return_nested_env: bool = False,
    ) -> TensorDictBase:
        """Executes a rollout in the environment and returns a flat or nested tensordict graph.

        The function will stop as soon as one of the contained environments
        returns :obj:`tensordict.get("done") == True`
        (unless :obj:`break_when_any_done` is turned off).

        Args:
            max_steps (int): maximum number of steps to be executed. The actual number of steps can be smaller if
                the environment reaches a done state before max_steps have been executed.
            policy (callable, optional): callable to be called to compute the desired action. If no policy is provided,
                actions will be called using :obj:`env.rand_step()`
                default = None
            callback (callable, optional): function to be called at each iteration with the given TensorDict.
            auto_reset (bool, optional): if True, resets automatically the environment
                if it is in a done state when the rollout is initiated.
                Default is :obj:`True`.
            auto_cast_to_device (bool, optional): if True, the device of the tensordict is automatically cast to the
                policy device before the policy is used. Default is :obj:`False`.
            break_when_any_done (bool): breaks if any of the done state is True. Default is True.
            return_contiguous (bool): if False, a LazyStackedTensorDict will be returned. Default is True.
            tensordict (TensorDict, optional): if auto_reset is False, an initial
                tensordict must be provided.
            exclude_private_keys (bool, optional): if True, keys with a :obj:`"_"` prefix
                are removed from the output. Default: :obj:`True`.
            return_nested_env (bool, optional): if True, a nested tensordict will be returned.
                Otherwise each step of the trajectory will be collected and stacked in a
                tensordict of size :obj:`torch.Size([max_steps])`. Default: False

        Returns:
            TensorDict object containing the resulting trajectory.

        """
        try:
            policy_device = next(policy.parameters()).device
        except AttributeError:
            policy_device = "cpu"

        env_device = self.device

        if auto_reset:
            if tensordict is not None:
                raise RuntimeError(
                    "tensordict cannot be provided when auto_reset is True"
                )
            tensordict = self.reset()
        elif tensordict is None:
            raise RuntimeError("tensordict must be provided when auto_reset is False")

        if policy is None:

            def policy(td):
                return td.set("action", self.action_spec.rand(self.batch_size))

        reset_tensordict = tensordict
        if not return_nested_env:
            tensordicts = []
        if not tensordict.get("prev_done"):
            for i in range(max_steps):
                if auto_cast_to_device:
                    tensordict = tensordict.to(policy_device)
                tensordict = policy(tensordict)
                if auto_cast_to_device:
                    tensordict = tensordict.to(env_device)
                # 2 options: either the action has already been tried, or we need to
                # compute it
                hashed_action = self._hash_action(tensordict["action"])
                if (
                    "_children" not in tensordict.keys()
                    or hashed_action not in tensordict.get("_children").keys()
                ):
                    tensordict = self.step(tensordict)
                if not return_nested_env:
                    if exclude_private_keys:
                        tensordicts.append(exclude_private(tensordict).clone())
                    else:
                        tensordicts.append(tensordict.clone())
                tensordict = tensordict["_children", hashed_action]

                if (
                    break_when_any_done and tensordict.get("prev_done").any()
                ) or i == max_steps - 1:
                    break

                if callback is not None:
                    callback(self, tensordict)
        else:
            raise Exception("reset env before calling rollout!")

        batch_size = self.batch_size if tensordict is None else tensordict.batch_size
        if return_nested_env:
            return reset_tensordict

        out_td = torch.stack(tensordicts, len(batch_size))
        if return_contiguous:
            return out_td.contiguous()
        return out_td
