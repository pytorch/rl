.. module:: torchrl.data

torchrl.data package
===================

Replay Buffers
--------------

Replay buffers are a central part of off-policy RL algorithms. TorchRL provides an efficient implementation of a few,
widely used replay buffers:

.. autoclass:: torchrl.data.ReplayBuffer
.. autoclass:: torchrl.data.PrioritizedReplayBuffer
.. autoclass:: torchrl.data.TensorDictReplayBuffer
.. autoclass:: torchrl.data.TensorDictPrioritizedReplayBuffer


TensorDict
----------

Passing data across objects can become a burdensome task when designing high-level classes: for instance it can be
hard to design an actor class that can take an arbitrary number of inputs and return an arbitrary number of inputs. The
`TensorDict` class simplifies this process by packing together a bag of tensors in a dictionary-like object. This
class supports a set of basic operations on tensors to facilitate the manipulation of entire batch of data (e.g.
`torch.cat`, `torch.stack`, `.to(device)` etc.).

.. autoclass:: torchrl.data.TensorDict

TensorSpec
----------

The `TensorSpec` parent class and subclasses define the basic properties of observations and actions in TorchRL, such
as shape, device, dtype and domain.

.. autoclass:: torchrl.data.TensorSpec
.. autoclass:: torchrl.data.BoundedTensorSpec
.. autoclass:: torchrl.data.OneHotDiscreteTensorSpec
.. autoclass:: torchrl.data.UnboundedContinuousTensorSpec
.. autoclass:: torchrl.data.NdBoundedTensorSpec
.. autoclass:: torchrl.data.NdUnboundedContinuousTensorSpec
.. autoclass:: torchrl.data.BinaryDiscreteTensorSpec
.. autoclass:: torchrl.data.MultOneHotDiscreteTensorSpec
.. autoclass:: torchrl.data.CompositeSpec

Transforms
----------

In most cases, the raw output of an environment must be treated before being passed to another object (such as a
policy or a value operator). To do this, TorchRL provides a set of transforms that aim at reproducing the transform
logic of `torch.distributions.Transform` and `torchvision.transforms`.

.. autoclass:: torchrl.data.Transform
.. autoclass:: torchrl.data.TransformedEnv
.. autoclass:: torchrl.data.Compose
.. autoclass:: torchrl.data.CatTensors
.. autoclass:: torchrl.data.CatFrames
.. autoclass:: torchrl.data.RewardClipping
.. autoclass:: torchrl.data.Resize
.. autoclass:: torchrl.data.GrayScale
.. autoclass:: torchrl.data.ToTensorImage
.. autoclass:: torchrl.data.ObservationNorm
.. autoclass:: torchrl.data.RewardScaling
.. autoclass:: torchrl.data.ObservationTransform
.. autoclass:: torchrl.data.FiniteTensorDictCheck
.. autoclass:: torchrl.data.DoubleToFloat
.. autoclass:: torchrl.data.NoopResetEnv
.. autoclass:: torchrl.data.BinerizeReward
.. autoclass:: torchrl.data.PinMemoryTransform
.. autoclass:: torchrl.data.VecNorm
