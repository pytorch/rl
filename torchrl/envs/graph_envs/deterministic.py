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
        """
        raise NotImplementedError

    def rollout(
        self,
        max_steps: int,
        policy: Optional[Callable[[TensorDictBase], TensorDictBase]] = None,
        callback: Optional[Callable[[TensorDictBase, ...], TensorDictBase]] = None,
        auto_reset: bool = True,
        auto_cast_to_device: bool = False,
        break_when_any_done: bool = True,
        return_contiguous: bool = True,
        tensordict: Optional[TensorDictBase] = None,
        exclude_private_keys: Optional[bool] = True,
    ) -> TensorDictBase:
        """Executes a rollout in the environment and returns a flat graph.

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

        tensordicts = []
        if not self.is_done:
            for i in range(max_steps):
                if auto_cast_to_device:
                    tensordict = tensordict.to(policy_device)
                tensordict = policy(tensordict)
                if auto_cast_to_device:
                    tensordict = tensordict.to(env_device)
                # 2 options: either the action has already been tried, or we need to
                # compute it
                hashed_action = self._hash_action(tensordict["action"])
                if hashed_action not in tensordict.get("_children").keys():
                    tensordict = self.step(tensordict)
                tensordict = tensordict["_children", hashed_action]
                if exclude_private_keys:
                    tensordicts.append(exclude_private(tensordict).clone())
                else:
                    tensordicts.append(tensordict.clone())

                if (
                    break_when_any_done and tensordict.get("done").any()
                ) or i == max_steps - 1:
                    break

                if callback is not None:
                    callback(self, tensordict)
        else:
            raise Exception("reset env before calling rollout!")

        batch_size = self.batch_size if tensordict is None else tensordict.batch_size

        out_td = torch.stack(tensordicts, len(batch_size))
        if return_contiguous:
            return out_td.contiguous()
        return out_td
