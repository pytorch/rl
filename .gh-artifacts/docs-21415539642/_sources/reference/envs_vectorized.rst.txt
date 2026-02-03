.. currentmodule:: torchrl.envs

Vectorized and Parallel Environments
====================================

Vectorized (or better: parallel) environments is a common feature in Reinforcement Learning
where executing the environment step can be cpu-intensive.
Some libraries such as `gym3 <https://github.com/openai/gym3>`_ or `EnvPool <https://github.com/sail-sg/envpool>`_
offer interfaces to execute batches of environments simultaneously.
While they often offer a very competitive computational advantage, they do not
necessarily scale to the wide variety of environment libraries supported by TorchRL.
Therefore, TorchRL offers its own, generic :class:`ParallelEnv` class to run multiple
environments in parallel.
As this class inherits from :class:`SerialEnv`, it enjoys the exact same API as other environment.
Of course, a :class:`ParallelEnv` will have a batch size that corresponds to its environment count:

.. note::
  Given the library's many optional dependencies (eg, Gym, Gymnasium, and many others)
  warnings can quickly become quite annoying in multiprocessed / distributed settings.
  By default, TorchRL filters out these warnings in sub-processes. If one still wishes to
  see these warnings, they can be displayed by setting ``torchrl.filter_warnings_subprocess=False``.

It is important that your environment specs match the input and output that it sends and receives, as
:class:`ParallelEnv` will create buffers from these specs to communicate with the spawn processes.
Check the :func:`~torchrl.envs.utils.check_env_specs` method for a sanity check.

.. code-block::
   :caption: Parallel environment

        >>> def make_env():
        ...     return GymEnv("Pendulum-v1", from_pixels=True, g=9.81, device="cuda:0")
        >>> check_env_specs(env)  # this must pass for ParallelEnv to work
        >>> env = ParallelEnv(4, make_env)
        >>> print(env.batch_size)
        torch.Size([4])

:class:`ParallelEnv` allows to retrieve the attributes from its contained environments:
one can simply call:

.. code-block::
   :caption: Parallel environment attributes

        >>> a, b, c, d = env.g  # gets the g-force of the various envs, which we set to 9.81 before
        >>> print(a)
        9.81

.. note::

  *A note on performance*: launching a :class:`~.ParallelEnv` can take quite some time
  as it requires to launch as many python instances as there are processes. Due to
  the time that it takes to run ``import torch`` (and other imports), starting the
  parallel env can be a bottleneck. This is why, for instance, TorchRL tests are so slow.
  Once the environment is launched, a great speedup should be observed.

.. note::

  *TorchRL requires precise specs*: Another thing to take in consideration is
  that :class:`ParallelEnv` (as well as data collectors)
  will create data buffers based on the environment specs to pass data from one process
  to another. This means that a misspecified spec (input, observation or reward) will
  cause a breakage at runtime as the data can't be written on the preallocated buffer.
  In general, an environment should be tested using the :func:`~.utils.check_env_specs`
  test function before being used in a :class:`ParallelEnv`. This function will raise
  an assertion error whenever the preallocated buffer and the collected data mismatch.

We also offer the :class:`~.SerialEnv` class that enjoys the exact same API but is executed
serially. This is mostly useful for testing purposes, when one wants to assess the
behavior of a :class:`~.ParallelEnv` without launching the subprocesses.

In addition to :class:`~.ParallelEnv`, which offers process-based parallelism, we also provide a way to create
multithreaded environments with :class:`~.MultiThreadedEnv`. This class uses `EnvPool <https://github.com/sail-sg/envpool>`_
library underneath, which allows for higher performance, but at the same time restricts flexibility - one can only
create environments implemented in ``EnvPool``. This covers many popular RL environments types (Atari, Classic Control,
etc.), but one can not use an arbitrary TorchRL environment, as it is possible with :class:`~.ParallelEnv`. Run
`benchmarks/benchmark_batched_envs.py` to compare performance of different ways to parallelize batched environments.

Vectorized environment classes
------------------------------

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    SerialEnv
    ParallelEnv
    EnvCreator

Partial steps and partial resets
--------------------------------

TorchRL allows environments to reset some but not all the environments, or run a step in one but not all environments.
If there is only one environment in the batch, then a partial reset / step is also allowed with the behavior detailed
below.

Batching environments and locking the batch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _ref_batch_locked:

Before detailing what partial resets and partial steps do, we must distinguish cases where an environment has
a batch size of its own (mostly stateful environments) or when the environment is just a mere module that, given an
input of arbitrary size, batches the operations over all elements (mostly stateless environments).

This is controlled via the :attr:`~torchrl.envs.batch_locked` attribute: a batch-locked environment requires all input
tensordicts to have the same batch-size as the env's. Typical examples of these environments are
:class:`~torchrl.envs.GymEnv` and related. Batch-unlocked envs are by contrast allowed to work with any input size.
Notable examples are :class:`~torchrl.envs.BraxEnv` or :class:`~torchrl.envs.JumanjiEnv`.

Executing partial steps in a batch-unlocked environment is straightforward: one just needs to mask the part of the
tensordict that does not need to be executed, pass the other part to `step` and merge the results with the previous
input.

Batched environments (:class:`~torchrl.envs.ParallelEnv` and :class:`~torchrl.envs.SerialEnv`) can also deal with
partial steps easily, they just pass the actions to the sub-environments that are required to be executed.

In all other cases, TorchRL assumes that the environment handles the partial steps correctly.

.. warning:: This means that custom environments may silently run the non-required steps as there is no way for torchrl
    to control what happens within the `_step` method!

Partial Steps
~~~~~~~~~~~~~

.. _ref_partial_steps:

Partial steps are controlled via the temporary key `"_step"` which points to a boolean mask of the
size of the tensordict that holds it. The classes armed to deal with this are:

- Batched environments: :class:`~torchrl.envs.ParallelEnv` and :class:`~torchrl.envs.SerialEnv` will dispatch the
  action to and only to the environments where `"_step"` is `True`;
- Batch-unlocked environments;
- Unbatched environments (i.e., environments without batch size). In these environments, the :meth:`~torchrl.envs.EnvBase.step`
  method will first look for a `"_step"` entry and, if present, act accordingly.
  If a :class:`~torchrl.envs.Transform` instance passes a `"_step"` entry to the tensordict, it is also captured by
  :class:`~torchrl.envs.TransformedEnv`'s own `_step` method which will skip the `base_env.step` as well as any further
  transformation.

When dealing with partial steps, the strategy is always to use the step output and mask missing values with the previous
content of the input tensordict, if present, or a `0`-valued tensor if the tensor cannot be found. This means that
if the input tensordict does not contain all the previous observations, then the output tensordict will be 0-valued for
all the non-stepped elements. Within batched environments, data collectors and rollouts utils, this is an issue that
is not observed because these classes handle the passing of data properly.

Partial steps are an essential feature of :meth:`~torchrl.envs.EnvBase.rollout` when `break_when_all_done` is `True`,
as the environments with a `True` done state will need to be skipped during calls to `_step`.

The :class:`~torchrl.envs.ConditionalSkip` transform allows you to programmatically ask for (partial) step skips.

Partial Resets
~~~~~~~~~~~~~~

.. _ref_partial_resets:

Partial resets work pretty much like partial steps, but with the `"_reset"` entry.

The same restrictions of partial steps apply to partial resets.

Likewise, partial resets are an essential feature of :meth:`~torchrl.envs.EnvBase.rollout` when `break_when_any_done` is `True`,
as the environments with a `True` done state will need to be reset, but not others.

See te following paragraph for a deep dive in partial resets within batched and vectorized environments.

Partial resets in detail
~~~~~~~~~~~~~~~~~~~~~~~~

TorchRL uses a private ``"_reset"`` key to indicate to the environment which
component (sub-environments or agents) should be reset.
This allows to reset some but not all of the components.

The ``"_reset"`` key has two distinct functionalities:

1. During a call to :meth:`~.EnvBase._reset`, the ``"_reset"`` key may or may
   not be present in the input tensordict. TorchRL's convention is that the
   absence of the ``"_reset"`` key at a given ``"done"`` level indicates
   a total reset of that level (unless a ``"_reset"`` key was found at a level
   above, see details below).
   If it is present, it is expected that those entries and only those components
   where the ``"_reset"`` entry is ``True`` (along key and shape dimension) will be reset.

   The way an environment deals with the ``"_reset"`` keys in its :meth:`~.EnvBase._reset`
   method is proper to its class.
   Designing an environment that behaves according to ``"_reset"`` inputs is the
   developer's responsibility, as TorchRL has no control over the inner logic
   of :meth:`~.EnvBase._reset`. Nevertheless, the following point should be
   kept in mind when designing that method.

2. After a call to :meth:`~.EnvBase._reset`, the output will be masked with the
   ``"_reset"`` entries and the output of the previous :meth:`~.EnvBase.step`
   will be written wherever the ``"_reset"`` was ``False``. In practice, this
   means that if a ``"_reset"`` modifies data that isn't exposed by it, this
   modification will be lost. After this masking operation, the ``"_reset"``
   entries will be erased from the :meth:`~.EnvBase.reset` outputs.

It must be pointed out that ``"_reset"`` is a private key, and it should only be
used when coding specific environment features that are internal facing.
In other words, this should NOT be used outside of the library, and developers
will keep the right to modify the logic of partial resets through ``"_reset"``
setting without preliminary warranty, as long as they don't affect TorchRL
internal tests.

Finally, the following assumptions are made and should be kept in mind when
designing reset functionalities:

- Each ``"_reset"`` is paired with a ``"done"`` entry (+ ``"terminated"`` and,
  possibly, ``"truncated"``). This means that the following structure is not
  allowed: ``TensorDict({"done": done, "nested": {"_reset": reset}}, [])``, as
  the ``"_reset"`` lives at a different nesting level than the ``"done"``.
- A reset at one level does not preclude the presence of a ``"_reset"`` at lower
  levels, but it annihilates its effects. The reason is simply that
  whether the ``"_reset"`` at the root level corresponds to an ``all()``, ``any()``
  or custom call to the nested ``"done"`` entries cannot be known in advance,
  and it is explicitly assumed that the ``"_reset"`` at the root was placed
  there to supersede the nested values (for an example, have a look at
  :class:`~.PettingZooWrapper` implementation where each group has one or more
  ``"done"`` entries associated which is aggregated at the root level with a
  ``any`` or ``all`` logic depending on the task).
- When calling :meth:`env.reset(tensordict)` with a partial ``"_reset"`` entry
  that will reset some but not all the done sub-environments, the input data
  should contain the data of the sub-environments that are __not__ being reset.
  The reason for this constrain lies in the fact that the output of the
  ``env._reset(data)`` can only be predicted for the entries that are reset.
  For the others, TorchRL cannot know in advance if they will be meaningful or
  not. For instance, one could perfectly just pad the values of the non-reset
  components, in which case the non-reset data will be meaningless and should
  be discarded.

Below, we give some examples of the expected effect that ``"_reset"`` keys will
have on an environment returning zeros after reset:

    >>> # single reset at the root
    >>> data = TensorDict({"val": [1, 1], "_reset": [False, True]}, [])
    >>> env.reset(data)
    >>> print(data.get("val"))  # only the second value is 0
    tensor([1, 0])
    >>> # nested resets
    >>> data = TensorDict({
    ...     ("agent0", "val"): [1, 1], ("agent0", "_reset"): [False, True],
    ...     ("agent1", "val"): [2, 2], ("agent1", "_reset"): [True, False],
    ... }, [])
    >>> env.reset(data)
    >>> print(data.get(("agent0", "val")))  # only the second value is 0
    tensor([1, 0])
    >>> print(data.get(("agent1", "val")))  # only the first value is 0
    tensor([0, 2])
    >>> # nested resets are overridden by a "_reset" at the root
    >>> data = TensorDict({
    ...     "_reset": [True, True],
    ...     ("agent0", "val"): [1, 1], ("agent0", "_reset"): [False, True],
    ...     ("agent1", "val"): [2, 2], ("agent1", "_reset"): [True, False],
    ... }, [])
    >>> env.reset(data)
    >>> print(data.get(("agent0", "val")))  # reset at the root overrides nested
    tensor([0, 0])
    >>> print(data.get(("agent1", "val")))  # reset at the root overrides nested
    tensor([0, 0])

.. code-block::
   :caption: Parallel environment reset

        >>> tensordict = TensorDict({"_reset": [[True], [False], [True], [True]]}, [4])
        >>> env.reset(tensordict)  # eliminates the "_reset" entry
        TensorDict(
            fields={
                terminated: Tensor(torch.Size([4, 1]), dtype=torch.bool),
                done: Tensor(torch.Size([4, 1]), dtype=torch.bool),
                pixels: Tensor(torch.Size([4, 500, 500, 3]), dtype=torch.uint8),
                truncated: Tensor(torch.Size([4, 1]), dtype=torch.bool),
            batch_size=torch.Size([4]),
            device=None,
            is_shared=True)

Async environments
------------------

Asynchronous environments allow for parallel execution of multiple environments, which can significantly speed up the
data collection process in reinforcement learning.

The `AsyncEnvPool` class and its subclasses provide a flexible interface for managing these environments using different
backends, such as threading and multiprocessing.

The `AsyncEnvPool` class serves as a base class for asynchronous environment pools, providing a common interface for
managing multiple environments concurrently. It supports different backends for parallel execution, such as threading
and multiprocessing, and provides methods for asynchronous stepping and resetting of environments.

Contrary to :class:`~torchrl.envs.ParallelEnv`, :class:`~torchrl.envs.AsyncEnvPool` and its subclasses permit the
execution of a given set of sub-environments while another task performed, allowing for complex asynchronous jobs to be
run at the same time. For instance, it is possible to execute some environments while the policy is running based on
the output of others.

This family of classes is particularly interesting when dealing with environments that have a high (and/or variable)
latency.

.. note:: This class and its subclasses should work when nested in with :class:`~torchrl.envs.TransformedEnv` and
    batched environments, but users won't currently be able to use the async features of the base environment when
    it's nested in these classes. One should prefer nested transformed envs within an `AsyncEnvPool` instead.
    If this is not possible, please raise an issue.

Classes
~~~~~~~

- :class:`~torchrl.envs.AsyncEnvPool`: A base class for asynchronous environment pools. It determines the backend
  implementation to use based on the provided arguments and manages the lifecycle of the environments.
- :class:`~torchrl.envs.ProcessorAsyncEnvPool`: An implementation of :class:`~torchrl.envs.AsyncEnvPool` using
  multiprocessing for parallel execution of environments. This class manages a pool of environments, each running in
  its own process, and provides methods for asynchronous stepping and resetting of environments using inter-process
  communication. It is automatically instantiated when `"multiprocessing"` is passed as a backend during the
  :class:`~torchrl.envs.AsyncEnvPool` instantiation.
- :class:`~torchrl.envs.ThreadingAsyncEnvPool`: An implementation of :class:`~torchrl.envs.AsyncEnvPool` using
  threading for parallel execution of environments. This class manages a pool of environments, each running in its own
  thread, and provides methods for asynchronous stepping and resetting of environments using a thread pool executor.
  It is automatically instantiated when `"threading"` is passed as a backend during the
  :class:`~torchrl.envs.AsyncEnvPool` instantiation.

Example
~~~~~~~

     >>> from functools import partial
     >>> from torchrl.envs import AsyncEnvPool, GymEnv
     >>> import torch
     >>> # Choose backend
     >>> backend = "threading"
     >>> env = AsyncEnvPool(
     >>>     [partial(GymEnv, "Pendulum-v1"), partial(GymEnv, "CartPole-v1")],
     >>>     stack="lazy",
     >>>     backend=backend
     >>> )
     >>> # Execute a synchronous reset
     >>> reset = env.reset()
     >>> print(reset)
     >>> # Execute a synchronous step
     >>> s = env.rand_step(reset)
     >>> print(s)
     >>> # Execute an asynchronous step in env 0
     >>> s0 = s[0]
     >>> s0["action"] = torch.randn(1).clamp(-1, 1)
     >>> s0["env_index"] = 0
     >>> env.async_step_send(s0)
     >>> # Receive data
     >>> s0_result = env.async_step_recv()
     >>> print('result', s0_result)
     >>> # Close env
     >>> env.close()


.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    AsyncEnvPool
    ProcessorAsyncEnvPool
    ThreadingAsyncEnvPool
