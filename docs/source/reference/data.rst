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
    Storage
    ListStorage
    LazyTensorStorage
    LazyMemmapStorage
    Writer
    RoundRobinWriter

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

Storing trajectories
~~~~~~~~~~~~~~~~~~~~

It is not too difficult to store trajectories in the replay buffer.
One element to pay attention to is that the size of the replay buffer is always
the size of the leading dimension of the storage: in other words, creating a
replay buffer with a storage of size 1M when storing multidimensional data
does not mean storing 1M frames but 1M trajectories.

When sampling trajectories, it may be desirable to sample sub-trajectories
to diversify learning or make the sampling more efficient.
To do this, we provide a custom :class:`~torchrl.envs.Transform` class named
:class:`~torchrl.envs.RandomCropTensorDict`. Here is an example of how this class
can be used:

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
    OpenMLExperienceReplay

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


Utils
-----

.. currentmodule:: torchrl.data

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    MultiStep
