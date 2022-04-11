.. environments:

Instantiating environments
==========================
TorchRL provides a collection of utilities to enable pytorch to interact with environments defined elsewhere.
The core idea of this module is to wrap an existing environment in a class that reads and writes tensordict objects.
The environment input (most likely an action) is read from a tensordict, possibly transformed and then passed to the
source environment. The same tensordict is then updated with the resulting set of outputs (state, observations, terminal
state indicator, reward or other).

An environment wrapper must contain the following methods:

- a `Wrapper._reset(self, tensordict)` method to reset the environment;
- a `Wrapper._step(self, tensordict)` method to execute a step in the environment;
- TODO: other?


TensorSpec
----------
A wrapper class should also take care of defining the environment features (observation and action spaces specifically).
There is little common ground between the existing libraries on this matter, which can make it difficult to write codes
that can accomodate more than one simulation library.
For this reason, TorchRL provides a `TensorSpec` abstract class that allows us to define the space of the tensors in a
flexible way. Besides the usual specificities that those class usually define (discrete or continuous, bounded or
unbounded, etc), `TensorSpec` also defines the device on which a tensor can be expected to be found or its
`torch.dtype`.

`TensorSpec` subclasses define the following methods:

- `TensorSpec.is_in(self, value)` asserts whether a value is compatible with the `TensorSpec` domain;
- `TensorSpec.rand(self, shape)` generates a random value in the tensorspec domain. If the domain is bounded or
discrete, this value is drawn uniformly in that space;
- `TensorSpec.project(self, value)` will attempt to project a tensor in the target domain if it is not contained in it.
The heuristic strategy used varies from case to case but in general it should be assumed that the resulting value will
mimimize the L1-distance between the tensor location and the boundaries;


Transforms
----------
Many existing libraries use environment wrappers to transform an environment output to the desired format. When more
than one transform is to be executed, a wrapper is placed around a wrapper and so forth.
This results in highly nested structures, such as `transformed_env = Wrapper1(Wrapper2(...(env)))`. If one desired to
access the environment, she must query the `env.env...env` attribute. Similarly, if an existing transform has to be
discared, the sequence of wrappers must be modified in a non-trivial way to accomodate for this change.
In TorchRL, we adopt a transform strategy that is more aligned with other PyTorch libraries, by placing the transforms
in an iterable object that gives access to each transform in a natural way:
`env = TrandsformedEnv(env, Compose(transform1, transform2, ...))`.
This allows us to have a clear representation of the chain of transforms.

As they act on `TensorDict` objects, each transform must be characterised by a sequence of input (and possibly output)
keys that point to the keys that will have to be read and modified by the transform. Here are a few examples:

.. code:: python

  from torchrl.envs.transforms import ToTensorImage, TransformedEnv
  env = make_env()
  transformed_env = TransformedEnv(
      env,
      ToTensorImage(in_keys=["observation_pixels"])) # assuming the output key is "observation_pixels"

A chain of transforms is easily defined through the `Compose` class:
.. code:: python

  transforms += [
      ToTensorImage(),  # transforms the default "next_observation_pixels" from W x H x 3 uint8 to a 3 x W x H float tensor
      Resize(84, 84),  # resize "next_observation_pixels" to 3 x 84 x 84
      GrayScale(),  # transforms the colours to grayscale
      CatFrames(keys=["next_observation_pixels"]),  # concatenates 4 following frames together
      ObservationNorm(loc=0.0, scale=1.0,
         standardize=True,
         keys=["next_observation_pixels"]),  # normalizes the observations
  ]
