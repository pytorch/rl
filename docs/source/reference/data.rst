.. currentmodule:: torchrl.data

torchrl.data package
====================

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

We also give users the ability to compose a replay buffer using the following components:

.. currentmodule:: torchrl.data.replay_buffers

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst


    Sampler
    PrioritizedSampler
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
    RoundRobinWriter
    TensorDictRoundRobinWriter
    TensorDictMaxValueWriter

Storage choice is very influential on replay buffer sampling latency, especially in distributed reinforcement learning settings with larger data volumes.
:class:`LazyMemmapStorage` is highly advised in distributed settings with shared storage due to the lower serialisation cost of MemmapTensors as well as the ability to specify file storage locations for improved node failure recovery.
The following mean sampling latency improvements over using ListStorage were found from rough benchmarking in https://github.com/pytorch/rl/tree/main/benchmarks/storage.

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
One element to pay attention to is that the size of the replay buffer is always
the size of the leading dimension of the storage: in other words, creating a
replay buffer with a storage of size 1M when storing multidimensional data
does not mean storing 1M frames but 1M trajectories. However, if trajectories
(or episodes/rollouts) are flattened before being stored, the capacity will still
be 1M steps.

When sampling trajectories, it may be desirable to sample sub-trajectories
to diversify learning or make the sampling more efficient.
TorchRL offers two distinctive ways of accomplishing this:
- The :class:`~torchrl.data.replay_buffers.samplers.SliceSampler` allows to
  sample a given number of slices of trajectories stored one after another
  along the leading dimension of the :class:`~torchrl.data.replay_buffers.samplers.TensorStorage`.
  This is the recommended way of sampling sub-trajectories in TorchRL __especially__
  when using offline datasets (which are stored using that convention).
  This strategy requires to flatten the trajectories before extending the replay
  buffer and reshaping them after sampling. The :class:`~torchrl.data.replay_buffers.samplers.SliceSampler`
  gives extensive details about this storage and sampling strategy.

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

Datasets
--------

TorchRL provides wrappers around offline RL datasets.
These data are presented a :class:`~torchrl.data.ReplayBuffer` instances, which
means that they can be customized at will with transforms, samplers and storages.
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

.. currentmodule:: torchrl.data.datasets

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst


    D4RLExperienceReplay
    MinariExperienceReplay
    OpenMLExperienceReplay
    RobosetExperienceReplay

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
