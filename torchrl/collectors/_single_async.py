from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule

from torchrl._utils import accept_remote_rref_udf_invocation
from torchrl.collectors._base import _make_legacy_metaclass
from torchrl.collectors._constants import DEFAULT_EXPLORATION_TYPE, ExplorationType
from torchrl.collectors._multi_async import MultiAsyncCollector
from torchrl.collectors._multi_base import _MultiCollectorMeta
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs import EnvBase


@accept_remote_rref_udf_invocation
class AsyncCollector(MultiAsyncCollector):
    """Runs a single DataCollector on a separate process.

    This is mostly useful for offline RL paradigms where the policy being
    trained can differ from the policy used to collect data. In online
    settings, a regular DataCollector should be preferred. This class is
    merely a wrapper around a MultiAsyncCollector where a single process
    is being created.

    Args:
        create_env_fn (Callabled): Callable returning an instance of EnvBase
        policy (Callable): Policy to be executed in the environment.
            Must accept :class:`tensordict.tensordict.TensorDictBase` object as input.
            If ``None`` is provided, the policy used will be a
            :class:`~torchrl.collectors.RandomPolicy` instance with the environment
            ``action_spec``.
            Accepted policies are usually subclasses of :class:`~tensordict.nn.TensorDictModuleBase`.
            This is the recommended usage of the collector.
            Other callables are accepted too:
            If the policy is not a ``TensorDictModuleBase`` (e.g., a regular :class:`~torch.nn.Module`
            instances) it will be wrapped in a `nn.Module` first.
            Then, the collector will try to assess if these
            modules require wrapping in a :class:`~tensordict.nn.TensorDictModule` or not.

            - If the policy forward signature matches any of ``forward(self, tensordict)``,
              ``forward(self, td)`` or ``forward(self, <anything>: TensorDictBase)`` (or
              any typing with a single argument typed as a subclass of ``TensorDictBase``)
              then the policy won't be wrapped in a :class:`~tensordict.nn.TensorDictModule`.

            - In all other cases an attempt to wrap it will be undergone as such: ``TensorDictModule(policy, in_keys=env_obs_key, out_keys=env.action_keys)``.

            .. note:: If the policy needs to be passed as a policy factory (e.g., in case it mustn't be serialized /
                pickled directly), the ``policy_factory`` should be used instead.

    Keyword Args:
        policy_factory (Callable[[], Callable], optional): a callable that returns
            a policy instance. This is exclusive with the `policy` argument.

            .. note:: `policy_factory` comes in handy whenever the policy cannot be serialized.

        frames_per_batch (int): A keyword-only argument representing the
            total number of elements in a batch.
        total_frames (int, optional): A keyword-only argument representing the
            total number of frames returned by the collector
            during its lifespan. If the ``total_frames`` is not divisible by
            ``frames_per_batch``, an exception is raised.
            Endless collectors can be created by passing ``total_frames=-1``.
            Defaults to ``-1`` (never ending collector).
        device (int, str or torch.device, optional): The generic device of the
            collector. The ``device`` args fills any non-specified device: if
            ``device`` is not ``None`` and any of ``storing_device``, ``policy_device`` or
            ``env_device`` is not specified, its value will be set to ``device``.
            Defaults to ``None`` (No default device).
            Supports a list of devices if one wishes to indicate a different device
            for each worker. The list must be as long as the number of workers.
        storing_device (int, str or torch.device, optional): The device on which
            the output :class:`~tensordict.TensorDict` will be stored.
            If ``device`` is passed and ``storing_device`` is ``None``, it will
            default to the value indicated by ``device``.
            For long trajectories, it may be necessary to store the data on a different
            device than the one where the policy and env are executed.
            Defaults to ``None`` (the output tensordict isn't on a specific device,
            leaf tensors sit on the device where they were created).
            Supports a list of devices if one wishes to indicate a different device
            for each worker. The list must be as long as the number of workers.
        env_device (int, str or torch.device, optional): The device on which
            the environment should be cast (or executed if that functionality is
            supported). If not specified and the env has a non-``None`` device,
            ``env_device`` will default to that value. If ``device`` is passed
            and ``env_device=None``, it will default to ``device``. If the value
            as such specified of ``env_device`` differs from ``policy_device``
            and one of them is not ``None``, the data will be cast to ``env_device``
            before being passed to the env (i.e., passing different devices to
            policy and env is supported). Defaults to ``None``.
            Supports a list of devices if one wishes to indicate a different device
            for each worker. The list must be as long as the number of workers.
        policy_device (int, str or torch.device, optional): The device on which
            the policy should be cast.
            If ``device`` is passed and ``policy_device=None``, it will default
            to ``device``. If the value as such specified of ``policy_device``
            differs from ``env_device`` and one of them is not ``None``,
            the data will be cast to ``policy_device`` before being passed to
            the policy (i.e., passing different devices to policy and env is
            supported). Defaults to ``None``.
            Supports a list of devices if one wishes to indicate a different device
            for each worker. The list must be as long as the number of workers.
        create_env_kwargs (dict, optional): A dictionary with the
            keyword arguments used to create an environment. If a list is
            provided, each of its elements will be assigned to a sub-collector.
        max_frames_per_traj (int, optional): Maximum steps per trajectory.
            Note that a trajectory can span across multiple batches (unless
            ``reset_at_each_iter`` is set to ``True``, see below).
            Once a trajectory reaches ``n_steps``, the environment is reset.
            If the environment wraps multiple environments together, the number
            of steps is tracked for each environment independently. Negative
            values are allowed, in which case this argument is ignored.
            Defaults to ``None`` (i.e. no maximum number of steps).
        init_random_frames (int, optional): Number of frames for which the
            policy is ignored before it is called. This feature is mainly
            intended to be used in offline/model-based settings, where a
            batch of random trajectories can be used to initialize training.
            If provided, it will be rounded up to the closest multiple of frames_per_batch.
            Defaults to ``None`` (i.e. no random frames).
        reset_at_each_iter (bool, optional): Whether environments should be reset
            at the beginning of a batch collection.
            Defaults to ``False``.
        postproc (Callable, optional): A post-processing transform, such as
            a :class:`~torchrl.envs.Transform` or a :class:`~torchrl.data.postprocs.MultiStep`
            instance.
            Defaults to ``None``.
        split_trajs (bool, optional): Boolean indicating whether the resulting
            TensorDict should be split according to the trajectories.
            See :func:`~torchrl.collectors.utils.split_trajectories` for more
            information.
            Defaults to ``False``.
        exploration_type (ExplorationType, optional): interaction mode to be used when
            collecting data. Must be one of ``torchrl.envs.utils.ExplorationType.DETERMINISTIC``,
            ``torchrl.envs.utils.ExplorationType.RANDOM``, ``torchrl.envs.utils.ExplorationType.MODE``
            or ``torchrl.envs.utils.ExplorationType.MEAN``.
        reset_when_done (bool, optional): if ``True`` (default), an environment
            that return a ``True`` value in its ``"done"`` or ``"truncated"``
            entry will be reset at the corresponding indices.
        update_at_each_batch (boolm optional): if ``True``, :meth:`update_policy_weights_()`
            will be called before (sync) or after (async) each data collection.
            Defaults to ``False``.
        preemptive_threshold (:obj:`float`, optional): a value between 0.0 and 1.0 that specifies the ratio of workers
            that will be allowed to finished collecting their rollout before the rest are forced to end early.
        num_threads (int, optional): number of threads for this process.
            Defaults to the number of workers.
        num_sub_threads (int, optional): number of threads of the subprocesses.
            Should be equal to one plus the number of processes launched within
            each subprocess (or one if a single process is launched).
            Defaults to 1 for safety: if none is indicated, launching multiple
            workers may charge the cpu load too much and harm performance.
        set_truncated (bool, optional): if ``True``, the truncated signals (and corresponding
            ``"done"`` but not ``"terminated"``) will be set to ``True`` when the last frame of
            a rollout is reached. If no ``"truncated"`` key is found, an exception is raised.
            Truncated keys can be set through ``env.add_truncated_keys``.
            Defaults to ``False``.
        track_policy_version (bool or PolicyVersion, optional): if ``True``, the collector will track the version of the policy.
            This will be mediated by the :class:`~torchrl.envs.llm.transforms.policy_version.PolicyVersion` transform, which will be added to the environment.
            Alternatively, a :class:`~torchrl.envs.llm.transforms.policy_version.PolicyVersion` instance can be passed, which will be used to track
            the policy version.
            Defaults to `False`.

    """

    def __init__(
        self,
        create_env_fn: Callable[[], EnvBase],
        policy: None
        | (TensorDictModule | Callable[[TensorDictBase], TensorDictBase]) = None,
        *,
        policy_factory: Callable[[], Callable] | None = None,
        frames_per_batch: int,
        total_frames: int | None = -1,
        device: DEVICE_TYPING | Sequence[DEVICE_TYPING] | None = None,
        storing_device: DEVICE_TYPING | Sequence[DEVICE_TYPING] | None = None,
        env_device: DEVICE_TYPING | Sequence[DEVICE_TYPING] | None = None,
        policy_device: DEVICE_TYPING | Sequence[DEVICE_TYPING] | None = None,
        create_env_kwargs: Sequence[dict[str, Any]] | None = None,
        max_frames_per_traj: int | None = None,
        init_random_frames: int | None = None,
        reset_at_each_iter: bool = False,
        postproc: Callable[[TensorDictBase], TensorDictBase] | None = None,
        split_trajs: bool | None = None,
        exploration_type: ExplorationType = DEFAULT_EXPLORATION_TYPE,
        reset_when_done: bool = True,
        update_at_each_batch: bool = False,
        preemptive_threshold: float | None = None,
        num_threads: int | None = None,
        num_sub_threads: int = 1,
        set_truncated: bool = False,
        track_policy_version: bool = False,
        **kwargs,
    ):
        super().__init__(
            create_env_fn=[create_env_fn],
            policy=policy,
            policy_factory=policy_factory,
            total_frames=total_frames,
            create_env_kwargs=[create_env_kwargs]
            if create_env_kwargs
            else create_env_kwargs,
            max_frames_per_traj=max_frames_per_traj,
            frames_per_batch=frames_per_batch,
            reset_at_each_iter=reset_at_each_iter,
            init_random_frames=init_random_frames,
            postproc=postproc,
            split_trajs=split_trajs,
            device=device,
            policy_device=policy_device,
            env_device=env_device,
            storing_device=storing_device,
            exploration_type=exploration_type,
            reset_when_done=reset_when_done,
            update_at_each_batch=update_at_each_batch,
            preemptive_threshold=preemptive_threshold,
            num_threads=num_threads,
            num_sub_threads=num_sub_threads,
            set_truncated=set_truncated,
            track_policy_version=track_policy_version,
            **kwargs,
        )

    # for RPC
    def next(self):
        return super().next()

    # for RPC
    def shutdown(
        self,
        timeout: float | None = None,
        close_env: bool = True,
        raise_on_error: bool = True,
    ) -> None:
        return super().shutdown(
            timeout=timeout, close_env=close_env, raise_on_error=raise_on_error
        )

    # for RPC
    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        return super().set_seed(seed, static_seed)

    # for RPC
    def state_dict(self) -> OrderedDict:
        return super().state_dict()

    # for RPC
    def load_state_dict(self, state_dict: OrderedDict) -> None:
        return super().load_state_dict(state_dict)


_LegacyAsyncCollectorMeta = _make_legacy_metaclass(_MultiCollectorMeta)


class aSyncDataCollector(AsyncCollector, metaclass=_LegacyAsyncCollectorMeta):
    """Deprecated version of :class:`~torchrl.collectors.AsyncCollector`."""

    ...
