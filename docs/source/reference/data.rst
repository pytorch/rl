.. currentmodule:: torchrl.data

torchrl.data package
====================

.. _ref_data:

Replay Buffers
--------------

Replay buffers are a central part of off-policy RL algorithms. TorchRL provides an efficient implementation of a few,
widely used replay buffers:


.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    ReplayBuffer
    PrioritizedReplayBuffer
    TensorDictReplayBuffer
    TensorDictPrioritizedReplayBuffer

Composable Replay Buffers
-------------------------

We also give users the ability to compose a replay buffer.
We provide a wide panel of solutions for replay buffer usage, including support for
almost any data type; storage in memory, on device or on physical memory;
several sampling strategies; usage of transforms etc.

Supported data types and choosing a storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In theory, replay buffers support any data type but we can't guarantee that each
component will support any data type. The most crude replay buffer implementation
is made of a :class:`~torchrl.data.replay_buffers.ReplayBuffer` base with a
:class:`~torchrl.data.replay_buffers.ListStorage` storage. This is very inefficient
but it will allow you to store complex data structures with non-tensor data.
Storages in contiguous memory include :class:`~torchrl.data.replay_buffers.TensorStorage`,
:class:`~torchrl.data.replay_buffers.LazyTensorStorage` and
:class:`~torchrl.data.replay_buffers.LazyMemmapStorage`.
These classes support :class:`~tensordict.TensorDict` data as first-class citizens, but also
any PyTree data structure (eg, tuples, lists, dictionaries and nested versions
of these). The :class:`~torchrl.data.replay_buffers.TensorStorage` storage requires
you to provide the storage at construction time, whereas :class:`~torchrl.data.replay_buffers.TensorStorage`
(RAM, CUDA) and :class:`~torchrl.data.replay_buffers.LazyMemmapStorage` (physical memory)
will preallocate the storage for you after they've been extended the first time.

Here are a few examples, starting with the generic :class:`~torchrl.data.replay_buffers.ListStorage`:

    >>> from torchrl.data.replay_buffers import ReplayBuffer, ListStorage
    >>> rb = ReplayBuffer(storage=ListStorage(10))
    >>> rb.add("a string!") # first element will be a string
    >>> rb.extend([30, None])  # element [1] is an int, [2] is None

Using a :class:`~torchrl.data.replay_buffers.TensorStorage` we tell our RB that
we want the storage to be contiguous, which is by far more efficient but also
more restrictive:

    >>> import torch
    >>> from torchrl.data.replay_buffers import ReplayBuffer, TensorStorage
    >>> container = torch.empty(10, 3, 64, 64, dtype=torch.unit8)
    >>> rb = ReplayBuffer(storage=TensorStorage(container))
    >>> img = torch.randint(255, (3, 64, 64), dtype=torch.uint8)
    >>> rb.add(img)

Next we can avoid creating the container and ask the storage to do it automatically.
This is very useful when using PyTrees and tensordicts! For PyTrees as other data
structures, :meth:`~torchrl.data.replay_buffers.ReplayBuffer.add` considers the sampled
passed to it as a single instance of the type. :meth:`~torchrl.data.replay_buffers.ReplayBuffer.extend`
on the other hand will consider that the data is an iterable. For tensors, tensordicts
and lists (see below), the iterable is looked for at the root level. For PyTrees,
we assume that the leading dimension of all the leaves (tensors) in the tree
match. If they don't, ``extend`` will throw an exception.

    >>> import torch
    >>> from tensordict import TensorDict
    >>> from torchrl.data.replay_buffers import ReplayBuffer, LazyMemmapStorage
    >>> rb_td = ReplayBuffer(storage=LazyMemmapStorage(10), batch_size=1)  # max 10 elements stored
    >>> rb_td.add(TensorDict({"img": torch.randint(255, (3, 64, 64), dtype=torch.unit8),
    ...     "labels": torch.randint(100, ())}, batch_size=[]))
    >>> rb_pytree = ReplayBuffer(storage=LazyMemmapStorage(10))  # max 10 elements stored
    >>> # extend with a PyTree where all tensors have the same leading dim (3)
    >>> rb_pytree.extend({"a": {"b": torch.randn(3), "c": [torch.zeros(3, 2), (torch.ones(3, 10),)]}})
    >>> assert len(rb_pytree) == 3  # the replay buffer has 3 elements!

.. note:: :meth:`~torchrl.data.replay_buffers.ReplayBuffer.extend` can have an
  ambiguous signature when dealing with lists of values, which should be interpreted
  either as PyTree (in which case all elements in the list will be put in a slice
  in the stored PyTree in the storage) or a list of values to add one at a time.
  To solve this, TorchRL makes the clear-cut distinction between list and tuple:
  a tuple will be viewed as a PyTree, a list (at the root level) will be interpreted
  as a stack of values to add one at a time to the buffer.

Sampling and indexing
~~~~~~~~~~~~~~~~~~~~~

Replay buffers can be indexed and sampled.
Indexing and sampling collect data at given indices in the storage and then process them
through a series of transforms and ``collate_fn`` that can be passed to the `__init__`
function of the replay buffer. ``collate_fn`` comes with default values that should
match user expectations in the majority of cases, such that you should not have
to worry about it most of the time. Transforms are usually instances of :class:`~torchrl.envs.transforms.Transform`
even though regular functions will work too (in the latter case, the :meth:`~torchrl.envs.transforms.Transform.inv`
method will obviously be ignored, whereas in the first case it can be used to
preprocess the data before it is passed to the buffer).
Finally, sampling can be achieved using multithreading by passing the number of threads
to the constructor through the ``prefetch`` keyword argument. We advise users to
benchmark this technique in real life settings before adopting it, as there is
no guarantee that it will lead to a faster throughput in practice depending on
the machine and setting where it is used.

When sampling, the ``batch_size`` can be either passed during construction
(e.g., if it's constant throughout training) or
to the :meth:`~torchrl.data.replay_buffers.ReplayBuffer.sample` method.

To further refine the sampling strategy, we advise you to look into our samplers!

Here are a couple of examples of how to get data out of a replay buffer:

    >>> first_elt = rb_td[0]
    >>> storage = rb_td[:] # returns all valid elements from the buffer
    >>> sample = rb_td.sample(128)
    >>> for data in rb_td:  # iterate over the buffer using the sampler -- batch-size was set in the constructor to 1
    ...     print(data)

using the following components:

.. currentmodule:: torchrl.data.replay_buffers

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst


    Sampler
    PrioritizedSampler
    PrioritizedSliceSampler
    RandomSampler
    SamplerWithoutReplacement
    SliceSampler
    SliceSamplerWithoutReplacement
    Storage
    ListStorage
    LazyTensorStorage
    LazyMemmapStorage
    TensorStorage
    Writer
    ImmutableDatasetWriter
    RoundRobinWriter
    TensorDictRoundRobinWriter
    TensorDictMaxValueWriter

Storage choice is very influential on replay buffer sampling latency, especially
in distributed reinforcement learning settings with larger data volumes.
:class:`~torchrl.data.replay_buffers.storages.LazyMemmapStorage` is highly
advised in distributed settings with shared storage due to the lower serialisation
cost of MemoryMappedTensors as well as the ability to specify file storage locations
for improved node failure recovery.
The following mean sampling latency improvements over using :class:`~torchrl.data.replay_buffers.ListStorage`
were found from rough benchmarking in https://github.com/pytorch/rl/tree/main/benchmarks/storage.

+-------------------------------+-----------+
| Storage Type                  | Speed up  |
|                               |           |
+===============================+===========+
| :class:`ListStorage`          | 1x        |
+-------------------------------+-----------+
| :class:`LazyTensorStorage`    | 1.83x     |
+-------------------------------+-----------+
| :class:`LazyMemmapStorage`    | 3.44x     |
+-------------------------------+-----------+

Sharing replay buffers across processes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replay buffers can be shared between processes as long as their components are
sharable. This feature allows for multiple processes to collect data and populate a shared
replay buffer collaboratively, rather than centralizing the data on the main process
which can incur some data transmission overhead.

Sharable storages include :class:`~torchrl.data.replay_buffers.storages.LazyMemmapStorage`
or any subclass of :class:`~torchrl.data.replay_buffers.storages.TensorStorage`
as long as they are instantiated and their content is stored as memory-mapped
tensors. Stateful writers such as :class:`~torchrl.data.replay_buffers.writers.TensorDictRoundRobinWriter`
are currently not sharable, and the same goes for stateful samplers such as
:class:`~torchrl.data.replay_buffers.samplers.PrioritizedSampler`.

A shared replay buffer can be read and extended on any process that has access
to it, as the following example shows:

  >>> from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
  >>> import torch
  >>> from torch import multiprocessing as mp
  >>> from tensordict import TensorDict
  >>>
  >>> def worker(rb):
  ...     # Updates the replay buffer with new data
  ...     td = TensorDict({"a": torch.ones(10)}, [10])
  ...     rb.extend(td)
  ...
  >>> if __name__ == "__main__":
  ...     rb = TensorDictReplayBuffer(storage=LazyMemmapStorage(21))
  ...     td = TensorDict({"a": torch.zeros(10)}, [10])
  ...     rb.extend(td)
  ...
  ...     proc = mp.Process(target=worker, args=(rb,))
  ...     proc.start()
  ...     proc.join()
  ...     # the replay buffer now has a length of 20, since the worker updated it
  ...     assert len(rb) == 20
  ...     assert (rb["_data", "a"][:10] == 0).all()  # data from main process
  ...     assert (rb["_data", "a"][10:20] == 1).all()  # data from remote process


Storing trajectories
~~~~~~~~~~~~~~~~~~~~

It is not too difficult to store trajectories in the replay buffer.
One element to pay attention to is that the size of the replay buffer is by default
the size of the leading dimension of the storage: in other words, creating a
replay buffer with a storage of size 1M when storing multidimensional data
does not mean storing 1M frames but 1M trajectories. However, if trajectories
(or episodes/rollouts) are flattened before being stored, the capacity will still
be 1M steps.

There is a way to circumvent this by telling the storage how many dimensions
it should take into account when saving data. This can be done through the ``ndim``
keyword argument which is accepted by all contiguous storages such as
:class:`~torchrl.data.replay_buffers.TensorStorage` and the likes. When a
multidimensional storage is passed to a buffer, the buffer will automatically
consider the last dimension as the "time" dimension, as it is conventional in
TorchRL. This can be overridden through the ``dim_extend`` keyword argument
in :class:`~torchrl.data.ReplayBuffer`.
This is the recommended way to save trajectories that are obtained through
:class:`~torchrl.envs.ParallelEnv` or its serial counterpart, as we will see
below.

When sampling trajectories, it may be desirable to sample sub-trajectories
to diversify learning or make the sampling more efficient.
TorchRL offers two distinctive ways of accomplishing this:

- The :class:`~torchrl.data.replay_buffers.samplers.SliceSampler` allows to
  sample a given number of slices of trajectories stored one after another
  along the leading dimension of the :class:`~torchrl.data.replay_buffers.samplers.TensorStorage`.
  This is the recommended way of sampling sub-trajectories in TorchRL __especially__
  when using offline datasets (which are stored using that convention).
  This strategy requires to flatten the trajectories before extending the replay
  buffer and reshaping them after sampling.
  The :class:`~torchrl.data.replay_buffers.samplers.SliceSampler` class docstrings
  gives extensive details about this storage and sampling strategy.
  Note that :class:`~torchrl.data.replay_buffers.samplers.SliceSampler`
  is compatible with multidimensional storages. The following examples show
  how to use this feature with and without flattening of the tensordict.
  In the first scenario, we are collecting data from a single environment. In
  that case, we are happy with a storage that concatenates the data coming in
  along the first dimension, since there will be no interruption introduced
  by the collection schedule:

        >>> from torchrl.envs import TransformedEnv, StepCounter, GymEnv
        >>> from torchrl.collectors import SyncDataCollector, RandomPolicy
        >>> from torchrl.data import ReplayBuffer, LazyTensorStorage, SliceSampler
        >>> env = TransformedEnv(GymEnv("CartPole-v1"), StepCounter())
        >>> collector = SyncDataCollector(env,
        ...     RandomPolicy(env.action_spec),
        ...     frames_per_batch=10, total_frames=-1)
        >>> rb = ReplayBuffer(
        ...     storage=LazyTensorStorage(100),
        ...     sampler=SliceSampler(num_slices=8, traj_key=("collector", "traj_ids"),
        ...         truncated_key=None, strict_length=False),
        ...     batch_size=64)
        >>> for i, data in enumerate(collector):
        ...     rb.extend(data)
        ...     if i == 10:
        ...         break
        >>> assert len(rb) == 100, len(rb)
        >>> print(rb[:]["next", "step_count"])
        tensor([[32],
                [33],
                [34],
                [35],
                [36],
                [37],
                [38],
                [39],
                [40],
                [41],
                [11],
                [12],
                [13],
                [14],
                [15],
                [16],
                [17],
                [...

  If there are more than one environment run in a batch, we could still store
  the data in the same buffer as before by calling ``data.reshape(-1)`` which
  will flatten the ``[B, T]`` size into ``[B * T]`` but that means that the
  trajectories of, say, the first environment of the batch will be interleaved
  by trajectories of the other environments, a scenario that ``SliceSampler``
  cannot handle. To solve this, we suggest to use the ``ndim`` argument in the
  storage constructor:

        >>> env = TransformedEnv(SerialEnv(2,
        ...     lambda: GymEnv("CartPole-v1")), StepCounter())
        >>> collector = SyncDataCollector(env,
        ...     RandomPolicy(env.action_spec),
        ...     frames_per_batch=1, total_frames=-1)
        >>> rb = ReplayBuffer(
        ...     storage=LazyTensorStorage(100, ndim=2),
        ...     sampler=SliceSampler(num_slices=8, traj_key=("collector", "traj_ids"),
        ...         truncated_key=None, strict_length=False),
        ...     batch_size=64)
        >>> for i, data in enumerate(collector):
        ...     rb.extend(data)
        ...     if i == 100:
        ...         break
        >>> assert len(rb) == 100, len(rb)
        >>> print(rb[:]["next", "step_count"].squeeze())
        tensor([[ 6,  5],
                [ 2,  2],
                [ 3,  3],
                [ 4,  4],
                [ 5,  5],
                [ 6,  6],
                [ 7,  7],
                [ 8,  8],
                [ 9,  9],
                [10, 10],
                [11, 11],
                [12, 12],
                [13, 13],
                [14, 14],
                [15, 15],
                [16, 16],
                [17, 17],
                [18,  1],
                [19,  2],
                [...


- Trajectories can also be stored independently, with the each element of the
  leading dimension pointing to a different trajectory. This requires
  for the trajectories to have a congruent shape (or to be padded).
  We provide a custom :class:`~torchrl.envs.Transform` class named
  :class:`~torchrl.envs.RandomCropTensorDict` that allows to sample
  sub-trajectories in the buffer. Note that, unlike the :class:`~torchrl.data.replay_buffers.samplers.SliceSampler`-based
  strategy, here having an ``"episode"`` or ``"done"`` key pointing at the
  start and stop signals isn't required.
  Here is an example of how this class can be used:

  .. code-block::Python

      >>> import torch
      >>> from tensordict import TensorDict
      >>> from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
      >>> from torchrl.envs import RandomCropTensorDict
      >>>
      >>> obs = torch.randn(100, 50, 1)
      >>> data = TensorDict({"obs": obs[:-1], "next": {"obs": obs[1:]}}, [99])
      >>> rb = TensorDictReplayBuffer(storage=LazyMemmapStorage(1000))
      >>> rb.extend(data)
      >>> # subsample trajectories of length 10
      >>> rb.append_transform(RandomCropTensorDict(sub_seq_len=10))
      >>> print(rb.sample(128))
      TensorDict(
          fields={
              index: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.int32, is_shared=False),
              next: TensorDict(
                  fields={
                      obs: Tensor(shape=torch.Size([10, 50, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                  batch_size=torch.Size([10]),
                  device=None,
                  is_shared=False),
              obs: Tensor(shape=torch.Size([10, 50, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
          batch_size=torch.Size([10]),
          device=None,
          is_shared=False)

Checkpointing Replay Buffers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each component of the replay buffer can potentially be stateful and, as such,
require a dedicated way of being serialized.
Our replay buffer enjoys two separate APIs for saving their state on disk:
:meth:`~torchrl.data.ReplayBuffer.dumps` and :meth:`~torchrl.data.ReplayBuffer.loads` will save the
data of each component except transforms (storage, writer, sampler) using memory-mapped
tensors and json files for the metadata. This will work across all classes except
:class:`~torchrl.data.replay_buffers.storages.ListStorage`, which content
cannot be anticipated (and as such does not comply with memory-mapped data
structures such as those that can be found in the tensordict library).
This API guarantees that a buffer that is saved and then loaded back will be in
the exact same state, whether we look at the status of its sampler (eg, priority trees)
its writer (eg, max writer heaps) or its storage.
Under the hood, :meth:`~torchrl.data.ReplayBuffer.dumps` will just call the public
`dumps` method in a specific folder for each of its components (except transforms
which we don't assume to be serializable using memory-mapped tensors in general).

Whenever saving data using :meth:`~torchrl.data.ReplayBuffer.dumps` is not possible, an
alternative way is to use :meth:`~torchrl.data.ReplayBuffer.state_dict`, which returns a data
structure that can be saved using :func:`torch.save` and loaded using :func:`torch.load`
before calling :meth:`~torchrl.data.ReplayBuffer.load_state_dict`. The drawback
of this method is that it will struggle to save big data structures, which is a
common setting when using replay buffers.

TorchRL Episode Data Format (TED)
---------------------------------

.. _TED-format:

In TorchRL, sequential data is consistently presented in a specific format, known
as the TorchRL Episode Data Format (TED). This format is crucial for the seamless
integration and functioning of various components within TorchRL.

Some components, such as replay buffers, are somewhat indifferent to the data
format. However, others, particularly environments, heavily depend on it for smooth operation.

Therefore, it's essential to understand the TED, its purpose, and how to interact
with it. This guide will provide a clear explanation of the TED, why it's used,
and how to effectively work with it.

The Rationale Behind TED
~~~~~~~~~~~~~~~~~~~~~~~~

Formatting sequential data can be a complex task, especially in the realm of
Reinforcement Learning (RL). As practitioners, we often encounter situations
where data is delivered at the reset time (though not always), and sometimes data
is provided or discarded at the final step of the trajectory.

This variability means that we can observe data of different lengths in a dataset,
and it's not always immediately clear how to match each time step across the
various elements of this dataset. Consider the following ambiguous dataset structure:

    >>> observation.shape
    [200, 3]
    >>> action.shape
    [199, 4]
    >>> info.shape
    [200, 3]

At first glance, it seems that the info and observation were delivered
together (one of each at reset + one of each at each step call), as suggested by
the action having one less element. However, if info has one less element, we
must assume that it was either omitted at reset time or not delivered or recorded
for the last step of the trajectory. Without proper documentation of the data
structure, it's impossible to determine which info corresponds to which time step.

Complicating matters further, some datasets provide inconsistent data formats,
where ``observations`` or ``infos`` are missing at the start or end of the
rollout, and this behavior is often not documented.
The primary aim of TED is to eliminate these ambiguities by providing a clear
and consistent data representation.

The structure of TED
~~~~~~~~~~~~~~~~~~~~

TED is built upon the canonical definition of a Markov Decision Process (MDP) in RL contexts.
At each step, an observation conditions an action that results in (1) a new
observation, (2) an indicator of task completion (terminated, truncated, done),
and (3) a reward signal.

Some elements may be missing (for example, the reward is optional in imitation
learning contexts), or additional information may be passed through a state or
info container. In some cases, additional information is required to get the
observation during a call to ``step`` (for instance, in stateless environment simulators). Furthermore,
in certain scenarios, an "action" (or any other data) cannot be represented as a
single tensor and needs to be organized differently. For example, in Multi-Agent RL
settings, actions, observations, rewards, and completion signals may be composite.

TED accommodates all these scenarios with a single, uniform, and unambiguous
format. We distinguish what happens at time step ``t`` and ``t+1`` by setting a
limit at the time the action is executed. In other words, everything that was
present before ``env.step`` was called belongs to ``t``, and everything that
comes after belongs to ``t+1``.

The general rule is that everything that belongs to time step ``t`` is stored
at the root of the tensordict, while everything that belongs to ``t+1`` is stored
in the ``"next"`` entry of the tensordict. Here's an example:

    >>> data = env.reset()
    >>> data = policy(data)
    >>> print(env.step(data))
    TensorDict(
        fields={
            action: Tensor(...),  # The action taken at time t
            done: Tensor(...),  # The done state when the action was taken (at reset)
            next: TensorDict(  # all of this content comes from the call to `step`
                fields={
                    done: Tensor(...),  # The done state after the action has been taken
                    observation: Tensor(...),  # The observation resulting from the action
                    reward: Tensor(...),  # The reward resulting from the action
                    terminated: Tensor(...),  # The terminated state after the action has been taken
                    truncated: Tensor(...),  # The truncated state after the action has been taken
                batch_size=torch.Size([]),
                device=cpu,
                is_shared=False),
            observation: Tensor(...),  # the observation at reset
            terminated: Tensor(...),  # the terminated at reset
            truncated: Tensor(...),  # the truncated at reset
        batch_size=torch.Size([]),
        device=cpu,
        is_shared=False)

During a rollout (either using :class:`~torchrl.envs.EnvBase` or
:class:`~torchrl.collectors.SyncDataCollector`), the content of the ``"next"``
tensordict is brought to the root through the :func:`~torchrl.envs.utils.step_mdp`
function when the agent resets its step count: ``t <- t+1``. You can read more
about the environment API :ref:`here <Environment-API>`.

In most cases, there is no `True`-valued ``"done"`` state at the root since any
done state will trigger a (partial) reset which will turn the ``"done"`` to ``False``.
However, this is only true as long as resets are automatically performed. In some
cases, partial resets will not trigger a reset, so we retain these data, which
should have a considerably lower memory footprint than observations, for instance.

This format eliminates any ambiguity regarding the matching of an observation with
its action, info, or done state.

Dimensionality of the Tensordict
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

During a rollout, all collected tensordicts will be stacked along a new dimension
positioned at the end. Both collectors and environments will label this dimension
with the ``"time"`` name. Here's an example:

    >>> rollout = env.rollout(10, policy)
    >>> assert rollout.shape[-1] == 10
    >>> assert rollout.names[-1] == "time"

This ensures that the time dimension is clearly marked and easily identifiable
in the data structure.

Special cases and footnotes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-Agent data presentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The multi-agent data formatting documentation can be accessed in the :ref:`MARL environment API <MARL-environment-API>` section.

Memory-based policies (RNNs and Transformers)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the examples provided above, only ``env.step(data)`` generates data that
needs to be read in the next step. However, in some cases, the policy also
outputs information that will be required in the next step. This is typically
the case for RNN-based policies, which output an action as well as a recurrent
state that needs to be used in the next step.
To accommodate this, we recommend users to adjust their RNN policy to write this
data under the ``"next"`` entry of the tensordict. This ensures that this content
will be brought to the root in the next step. More information can be found in
:class:`~torchrl.modules.GRUModule` and :class:`~torchrl.modules.LSTMModule`.

Multi-step
^^^^^^^^^^

Collectors allow users to skip steps when reading the data, accumulating reward
for the upcoming n steps. This technique is popular in DQN-like algorithms like Rainbow.
The :class:`~torchrl.data.postprocs.MultiStep` class performs this data transformation
on batches coming out of collectors. In these cases, a check like the following
will fail since the next observation is shifted by n steps:

    >>> assert (data[..., 1:]["observation"] == data[..., :-1]["next", "observation"]).all()

What about memory requirements?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Implemented naively, this data format consumes approximately twice the memory
that a flat representation would. In some memory-intensive settings
(for example, in the :class:`~torchrl.data.datasets.AtariDQNExperienceReplay` dataset),
we store only the ``T+1`` observation on disk and perform the formatting online at get time.
In other cases, we assume that the 2x memory cost is a small price to pay for a
clearer representation. However, generalizing the lazy representation for offline
datasets would certainly be a beneficial feature to have, and we welcome
contributions in this direction!

Datasets
--------

TorchRL provides wrappers around offline RL datasets.
These data are presented as :class:`~torchrl.data.ReplayBuffer` instances, which
means that they can be customized at will with transforms, samplers and storages.
For instance, entries can be filtered in or out of a dataset with :class:`~torchrl.envs.SelectTransform`
or :class:`~torchrl.envs.ExcludeTransform`.

By default, datasets are stored as memory mapped tensors, allowing them to be
promptly sampled with virtually no memory footprint.

Here's an example:

.. code::Python

  >>> from torchrl.data.datasets import D4RLExperienceReplay
  >>> from torchrl.data.replay_buffers import SamplerWithoutReplacement
  >>> from torchrl.envs.transforms import RenameTransform
  >>> dataset = D4RLExperienceReplay('kitchen-complete-v0', split_trajs=True, batch_size=10)
  >>> print(dataset.sample())  # will sample 10 trajectories since split_trajs is set to True
  TensorDict(
      fields={
          action: Tensor(shape=torch.Size([10, 207, 9]), device=cpu, dtype=torch.float32, is_shared=False),
          done: Tensor(shape=torch.Size([10, 207]), device=cpu, dtype=torch.bool, is_shared=False),
          index: Tensor(shape=torch.Size([10, 207]), device=cpu, dtype=torch.int32, is_shared=False),
          infos: Tensor(shape=torch.Size([10, 207]), device=cpu, dtype=torch.int64, is_shared=False),
          mask: Tensor(shape=torch.Size([10, 207]), device=cpu, dtype=torch.bool, is_shared=False),
          next: TensorDict(
              fields={
                  done: Tensor(shape=torch.Size([10, 207]), device=cpu, dtype=torch.bool, is_shared=False),
                  observation: Tensor(shape=torch.Size([10, 207, 60]), device=cpu, dtype=torch.float32, is_shared=False),
                  reward: Tensor(shape=torch.Size([10, 207]), device=cpu, dtype=torch.float32, is_shared=False)},
              batch_size=torch.Size([10, 207]),
              device=cpu,
              is_shared=False),
          observation: Tensor(shape=torch.Size([10, 207, 60]), device=cpu, dtype=torch.float32, is_shared=False),
          timeouts: Tensor(shape=torch.Size([10, 207]), device=cpu, dtype=torch.bool, is_shared=False),
          traj_ids: Tensor(shape=torch.Size([10, 207]), device=cpu, dtype=torch.int64, is_shared=False)},
      batch_size=torch.Size([10, 207]),
      device=cpu,
      is_shared=False)
  >>> dataset.append_transform(RenameTransform(["done", ("next", "done")], ["terminal", ("next", "terminal")]))
  >>> print(dataset.sample())  # The "done" has been renamed to "terminal"
  TensorDict(
      fields={
          action: Tensor(shape=torch.Size([10, 207, 9]), device=cpu, dtype=torch.float32, is_shared=False),
          terminal: Tensor(shape=torch.Size([10, 207]), device=cpu, dtype=torch.bool, is_shared=False),
          index: Tensor(shape=torch.Size([10, 207]), device=cpu, dtype=torch.int32, is_shared=False),
          infos: Tensor(shape=torch.Size([10, 207]), device=cpu, dtype=torch.int64, is_shared=False),
          mask: Tensor(shape=torch.Size([10, 207]), device=cpu, dtype=torch.bool, is_shared=False),
          next: TensorDict(
              fields={
                  terminal: Tensor(shape=torch.Size([10, 207]), device=cpu, dtype=torch.bool, is_shared=False),
                  observation: Tensor(shape=torch.Size([10, 207, 60]), device=cpu, dtype=torch.float32, is_shared=False),
                  reward: Tensor(shape=torch.Size([10, 207]), device=cpu, dtype=torch.float32, is_shared=False)},
              batch_size=torch.Size([10, 207]),
              device=cpu,
              is_shared=False),
          observation: Tensor(shape=torch.Size([10, 207, 60]), device=cpu, dtype=torch.float32, is_shared=False),
          timeouts: Tensor(shape=torch.Size([10, 207]), device=cpu, dtype=torch.bool, is_shared=False),
          traj_ids: Tensor(shape=torch.Size([10, 207]), device=cpu, dtype=torch.int64, is_shared=False)},
      batch_size=torch.Size([10, 207]),
      device=cpu,
      is_shared=False)
  >>> # we can also use a `SamplerWithoutReplacement` to iterate over the dataset with random samples:
  >>> dataset = D4RLExperienceReplay(
  ...   'kitchen-complete-v0',
  ...   sampler=SamplerWithoutReplacement(drop_last=True),
  ...   split_trajs=True,
  ...   batch_size=3)
  >>> for data in dataset:
  ...    print(data)
  ...

.. note::

  Installing dependencies is the responsibility of the user. For D4RL, a clone of
  `the repository <https://github.com/Farama-Foundation/D4RL>`_ is needed as
  the latest wheels are not published on PyPI. For OpenML, `scikit-learn <https://pypi.org/project/scikit-learn/>`_ and
  `pandas <https://pypi.org/project/pandas>`_ are required.

Transforming datasets
~~~~~~~~~~~~~~~~~~~~~

In many instances, the raw data isn't going to be used as-is.
The natural solution could be to pass a :class:`~torchrl.envs.transforms.Transform`
instance to the dataset constructor and modify the sample on-the-fly. This will
work but it will incur an extra runtime for the transform.
If the transformations can be (at least a part) pre-applied to the dataset,
a conisderable disk space and some incurred overhead at sampling time can be
saved. To do this, the
:meth:`~torchrl.data.datasets.BaseDatasetExperienceReplay.preprocess` can be
used. This method will run a per-sample preprocessing pipeline on each element
of the dataset, and replace the existing dataset by its transformed version.

Once transformed, re-creating the same dataset will produce another object with
the same transformed storage (unless ``download="force"`` is being used):

    >>> dataset = RobosetExperienceReplay(
    ...     "FK1-v4(expert)/FK1_MicroOpenRandom_v2d-v4", batch_size=32, download="force"
    ... )
    >>>
    >>> def func(data):
    ...     return data.set("obs_norm", data.get("observation").norm(dim=-1))
    ...
    >>> dataset.preprocess(
    ...     func,
    ...     num_workers=max(1, os.cpu_count() - 2),
    ...     num_chunks=1000,
    ...     mp_start_method="fork",
    ... )
    >>> sample = dataset.sample()
    >>> assert "obs_norm" in sample.keys()
    >>> # re-recreating the dataset gives us the transformed version back.
    >>> dataset = RobosetExperienceReplay(
    ...     "FK1-v4(expert)/FK1_MicroOpenRandom_v2d-v4", batch_size=32
    ... )
    >>> sample = dataset.sample()
    >>> assert "obs_norm" in sample.keys()


.. currentmodule:: torchrl.data.datasets

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    BaseDatasetExperienceReplay
    AtariDQNExperienceReplay
    D4RLExperienceReplay
    GenDGRLExperienceReplay
    MinariExperienceReplay
    OpenMLExperienceReplay
    OpenXExperienceReplay
    RobosetExperienceReplay
    VD4RLExperienceReplay

Composing datasets
~~~~~~~~~~~~~~~~~~

In offline RL, it is customary to work with more than one dataset at the same time.
Moreover, TorchRL usually has a fine-grained dataset nomenclature, where
each task is represented separately when other libraries will represent these
datasets in a more compact way. To allow users to compose multiple datasets
together, we propose a :class:`~torchrl.data.replay_buffers.ReplayBufferEnsemble`
primitive that allows users to sample from multiple datasets at once.

If the individual dataset formats differ, :class:`~torchrl.envs.Transform` instances
can be used. In the following example, we create two dummy datasets with semantically
identical entries that differ in names (``("some", "key")`` and ``"another_key"``)
and show how they can be renamed to have a matching name. We also resize images
such that they can be stacked together during sampling.

    >>> from torchrl.envs import Comopse, ToTensorImage, Resize, RenameTransform
    >>> from torchrl.data import TensorDictReplayBuffer, ReplayBufferEnsemble, LazyMemmapStorage
    >>> from tensordict import TensorDict
    >>> import torch
    >>> rb0 = TensorDictReplayBuffer(
    ...     storage=LazyMemmapStorage(10),
    ...     transform=Compose(
    ...         ToTensorImage(in_keys=["pixels", ("next", "pixels")]),
    ...         Resize(32, in_keys=["pixels", ("next", "pixels")]),
    ...         RenameTransform([("some", "key")], ["renamed"]),
    ...     ),
    ... )
    >>> rb1 = TensorDictReplayBuffer(
    ...     storage=LazyMemmapStorage(10),
    ...     transform=Compose(
    ...         ToTensorImage(in_keys=["pixels", ("next", "pixels")]),
    ...         Resize(32, in_keys=["pixels", ("next", "pixels")]),
    ...         RenameTransform(["another_key"], ["renamed"]),
    ...     ),
    ... )
    >>> rb = ReplayBufferEnsemble(
    ...     rb0,
    ...     rb1,
    ...     p=[0.5, 0.5],
    ...     transform=Resize(33, in_keys=["pixels"], out_keys=["pixels33"]),
    ... )
    >>> data0 = TensorDict(
    ...     {
    ...         "pixels": torch.randint(255, (10, 244, 244, 3)),
    ...         ("next", "pixels"): torch.randint(255, (10, 244, 244, 3)),
    ...         ("some", "key"): torch.randn(10),
    ...     },
    ...     batch_size=[10],
    ... )
    >>> data1 = TensorDict(
    ...     {
    ...         "pixels": torch.randint(255, (10, 64, 64, 3)),
    ...         ("next", "pixels"): torch.randint(255, (10, 64, 64, 3)),
    ...         "another_key": torch.randn(10),
    ...     },
    ...     batch_size=[10],
    ... )
    >>> rb[0].extend(data0)
    >>> rb[1].extend(data1)
    >>> for _ in range(2):
    ...     sample = rb.sample(10)
    ...     assert sample["next", "pixels"].shape == torch.Size([2, 5, 3, 32, 32])
    ...     assert sample["pixels"].shape == torch.Size([2, 5, 3, 32, 32])
    ...     assert sample["pixels33"].shape == torch.Size([2, 5, 3, 33, 33])
    ...     assert sample["renamed"].shape == torch.Size([2, 5])

.. currentmodule:: torchrl.data.replay_buffers


.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    ReplayBufferEnsemble
    SamplerEnsemble
    StorageEnsemble
    WriterEnsemble

TensorSpec
----------

The `TensorSpec` parent class and subclasses define the basic properties of observations and actions in TorchRL, such
as shape, device, dtype and domain.
It is important that your environment specs match the input and output that it sends and receives, as
:obj:`ParallelEnv` will create buffers from these specs to communicate with the spawn processes.
Check the :obj:`torchrl.envs.utils.check_env_specs` method for a sanity check.

.. currentmodule:: torchrl.data

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    TensorSpec
    BinaryDiscreteTensorSpec
    BoundedTensorSpec
    CompositeSpec
    DiscreteTensorSpec
    MultiDiscreteTensorSpec
    MultiOneHotDiscreteTensorSpec
    OneHotDiscreteTensorSpec
    UnboundedContinuousTensorSpec
    UnboundedDiscreteTensorSpec
    LazyStackedTensorSpec
    LazyStackedCompositeSpec

Reinforcement Learning From Human Feedback (RLHF)
-------------------------------------------------

Data is of utmost importance in Reinforcement Learning from Human Feedback (RLHF).
Given that these techniques are commonly employed in the realm of language,
which is scarcely addressed in other subdomains of RL within the library,
we offer specific utilities to facilitate interaction with external libraries
like datasets. These utilities consist of tools for tokenizing data, formatting
it in a manner suitable for TorchRL modules, and optimizing storage for
efficient sampling.

.. currentmodule:: torchrl.data

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    PairwiseDataset
    PromptData
    PromptTensorDictTokenizer
    RewardData
    RolloutFromModel
    TensorDictTokenizer
    TokenizedDatasetLoader
    create_infinite_iterator
    get_dataloader
    ConstantKLController
    AdaptiveKLController


Utils
-----

.. currentmodule:: torchrl.data

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    MultiStep
    consolidate_spec
    check_no_exclusive_keys
    contains_lazy_spec

.. currentmodule:: torchrl.envs.transforms.rb_transforms

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    MultiStepTransform
