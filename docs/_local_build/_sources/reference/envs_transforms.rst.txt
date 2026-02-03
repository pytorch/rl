.. currentmodule:: torchrl.envs.transforms

.. _transforms:

Transforms
==========

In most cases, the raw output of an environment must be treated before being passed to another object (such as a
policy or a value operator). To do this, TorchRL provides a set of transforms that aim at reproducing the transform
logic of `torch.distributions.Transform` and `torchvision.transforms`.
Our environment :ref:`tutorial <pendulum_tuto>`
provides more information on how to design a custom transform.

Transformed environments are build using the :class:`TransformedEnv` primitive.
Composed transforms are built using the :class:`Compose` class:

.. code-block::
   :caption: Transformed environment

        >>> base_env = GymEnv("Pendulum-v1", from_pixels=True, device="cuda:0")
        >>> transform = Compose(ToTensorImage(in_keys=["pixels"]), Resize(64, 64, in_keys=["pixels"]))
        >>> env = TransformedEnv(base_env, transform)

Transforms are usually subclasses of :class:`~torchrl.envs.transforms.Transform`, although any
``Callable[[TensorDictBase], TensorDictBase]``.

By default, the transformed environment will inherit the device of the
``base_env`` that is passed to it. The transforms will then be executed on that device.
It is now apparent that this can bring a significant speedup depending on the kind of
operations that is to be computed.

A great advantage of environment wrappers is that one can consult the environment up to that wrapper.
The same can be achieved with TorchRL transformed environments: the ``parent`` attribute will
return a new :class:`TransformedEnv` with all the transforms up to the transform of interest.
Re-using the example above:

.. code-block::
   :caption: Transform parent

        >>> resize_parent = env.transform[-1].parent  # returns the same as TransformedEnv(base_env, transform[:-1])


Transformed environment can be used with vectorized environments.
Since each transform uses a ``"in_keys"``/``"out_keys"`` set of keyword argument, it is
also easy to root the transform graph to each component of the observation data (e.g.
pixels or states etc).

Forward and inverse transforms
------------------------------

Transforms also have an :meth:`~torchrl.envs.Transform.inv` method that is called before the action is applied in reverse
order over the composed transform chain. This allows applying transforms to data in the environment before the action is
taken in the environment. The keys to be included in this inverse transform are passed through the `"in_keys_inv"`
keyword argument, and the out-keys default to these values in most cases:

.. code-block::
   :caption: Inverse transform

        >>> env.append_transform(DoubleToFloat(in_keys_inv=["action"]))  # will map the action from float32 to float64 before calling the base_env.step

The following paragraphs detail how one can think about what is to be considered `in_` or `out_` features.

Understanding Transform Keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In transforms, `in_keys` and `out_keys` define the interaction between the base environment and the outside world
(e.g., your policy):

- `in_keys` refers to the base environment's perspective (inner = `base_env` of the
  :class:`~torchrl.envs.TransformedEnv`).
- `out_keys` refers to the outside world (outer = `policy`, `agent`, etc.).

For example, with `in_keys=["obs"]` and `out_keys=["obs_standardized"]`, the policy will "see" a standardized
observation, while the base environment outputs a regular observation.

Similarly, for inverse keys:

- `in_keys_inv` refers to entries as seen by the base environment.
- `out_keys_inv` refers to entries as seen or produced by the policy.

The following figure illustrates this concept for the :class:`~torchrl.envs.RenameTransform` class: the input
`TensorDict` of the `step` function must include the `out_keys_inv` as they are part of the outside world. The
transform changes these names to match the names of the inner, base environment using the `in_keys_inv`.
The inverse process is executed with the output tensordict, where the `in_keys` are mapped to the corresponding
`out_keys`.

.. figure:: /_static/img/rename_transform.png

   Rename transform logic

.. note:: During a call to `inv`, the transforms are executed in reversed order (compared to the forward / step mode).

Transforming Tensors and Specs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When transforming actual tensors (coming from the policy), the process is schematically represented as:

    >>> for t in reversed(self.transform):
    ...     td = t.inv(td)

This starts with the outermost transform to the innermost transform, ensuring the action value exposed to the policy
is properly transformed.

For transforming the action spec, the process should go from innermost to outermost (similar to observation specs):

    >>> def transform_action_spec(self, action_spec):
    ...     for t in self.transform:
    ...         action_spec = t.transform_action_spec(action_spec)
    ...     return action_spec

A pseudocode for a single transform_action_spec could be:

    >>> def transform_action_spec(self, action_spec):
    ...    return spec_from_random_values(self._apply_transform(action_spec.rand()))

This approach ensures that the "outside" spec is inferred from the "inside" spec. Note that we did not call
`_inv_apply_transform` but `_apply_transform` on purpose!

Exposing Specs to the Outside World
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`TransformedEnv` will expose the specs corresponding to the `out_keys_inv` for actions and states.
For example, with :class:`~torchrl.envs.ActionDiscretizer`, the environment's action (e.g., `"action"`) is a float-valued
tensor that should not be generated when using :meth:`~torchrl.envs.EnvBase.rand_action` with the transformed
environment. Instead, `"action_discrete"` should be generated, and its continuous counterpart obtained from the
transform. Therefore, the user should see the `"action_discrete"` entry being exposed, but not `"action"`.

Designing your own Transform
----------------------------

To create a basic, custom transform, you need to subclass the `Transform` class and implement the
:meth:`~torchrl.envs._apply_transform` method. Here's an example of a simple transform that adds 1 to the observation
tensor:

    >>> class AddOneToObs(Transform):
    ...     """A transform that adds 1 to the observation tensor."""
    ...
    ...     def __init__(self):
    ...         super().__init__(in_keys=["observation"], out_keys=["observation"])
    ...
    ...     def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
    ...         return obs + 1


Tips for subclassing `Transform`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are various ways of subclassing a transform. The things to take into considerations are:

- Is the transform identical for each tensor / item being transformed? Use
  :meth:`~torchrl.envs.Transform._apply_transform` and :meth:`~torchrl.envs.Transform._inv_apply_transform`.
- The transform needs access to the input data to env.step as well as output? Rewrite
  :meth:`~torchrl.envs.Transform._step`.
  Otherwise, rewrite :meth:`~torchrl.envs.Transform._call` (or :meth:`~torchrl.envs.Transform._inv_call`).
- Is the transform to be used within a replay buffer? Overwrite :meth:`~torchrl.envs.Transform.forward`,
  :meth:`~torchrl.envs.Transform.inv`, :meth:`~torchrl.envs.Transform._apply_transform` or
  :meth:`~torchrl.envs.Transform._inv_apply_transform`.
- Within a transform, you can access (and make calls to) the parent environment using
  :attr:`~torchrl.envs.Transform.parent` (the base env + all transforms till this one) or
  :meth:`~torchrl.envs.Transform.container` (The object that encapsulates the transform).
- Don't forget to edits the specs if needed: top level: :meth:`~torchrl.envs.Transform.transform_output_spec`,
  :meth:`~torchrl.envs.Transform.transform_input_spec`.
  Leaf level: :meth:`~torchrl.envs.Transform.transform_observation_spec`,
  :meth:`~torchrl.envs.Transform.transform_action_spec`, :meth:`~torchrl.envs.Transform.transform_state_spec`,
  :meth:`~torchrl.envs.Transform.transform_reward_spec` and
  :meth:`~torchrl.envs.Transform.transform_reward_spec`.

For practical examples, see the methods listed above.

You can use a transform in an environment by passing it to the TransformedEnv constructor:

    >>> env = TransformedEnv(GymEnv("Pendulum-v1"), AddOneToObs())

You can compose multiple transforms together using the Compose class:

    >>> transform = Compose(AddOneToObs(), RewardSum())
    >>> env = TransformedEnv(GymEnv("Pendulum-v1"), transform)

Inverse Transforms
~~~~~~~~~~~~~~~~~~

Some transforms have an inverse transform that can be used to undo the transformation. For example, the AddOneToAction
transform has an inverse transform that subtracts 1 from the action tensor:

    >>> class AddOneToAction(Transform):
    ...     """A transform that adds 1 to the action tensor."""
    ...     def __init__(self):
    ...         super().__init__(in_keys=[], out_keys=[], in_keys_inv=["action"], out_keys_inv=["action"])
    ...     def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
    ...         return action + 1

Using a Transform with a Replay Buffer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use a transform with a replay buffer by passing it to the ReplayBuffer constructor:

Cloning transforms
~~~~~~~~~~~~~~~~~~

Because transforms appended to an environment are "registered" to this environment
through the ``transform.parent`` property, when manipulating transforms we should keep
in mind that the parent may come and go following what is being done with the transform.
Here are some examples: if we get a single transform from a :class:`Compose` object,
this transform will keep its parent:

    >>> third_transform = env.transform[2]
    >>> assert third_transform.parent is not None

This means that using this transform for another environment is prohibited, as
the other environment would replace the parent and this may lead to unexpected
behviours. Fortunately, the :class:`Transform` class comes with a :func:`clone`
method that will erase the parent while keeping the identity of all the
registered buffers:

    >>> TransformedEnv(base_env, third_transform)  # raises an Exception as third_transform already has a parent
    >>> TransformedEnv(base_env, third_transform.clone())  # works

On a single process or if the buffers are placed in shared memory, this will
result in all the clone transforms to keep the same behavior even if the
buffers are changed in place (which is what will happen with the :class:`CatFrames`
transform, for instance). In distributed settings, this may not hold and one
should be careful about the expected behavior of the cloned transforms in this
context.
Finally, notice that indexing multiple transforms from a :class:`Compose` transform
may also result in loss of parenthood for these transforms: the reason is that
indexing a :class:`Compose` transform results in another :class:`Compose` transform
that does not have a parent environment. Hence, we have to clone the sub-transforms
to be able to create this other composition:

    >>> env = TransformedEnv(base_env, Compose(transform1, transform2, transform3))
    >>> last_two = env.transform[-2:]
    >>> assert isinstance(last_two, Compose)
    >>> assert last_two.parent is None
    >>> assert last_two[0] is not transform2
    >>> assert isinstance(last_two[0], type(transform2))  # and the buffers will match
    >>> assert last_two[1] is not transform3
    >>> assert isinstance(last_two[1], type(transform3))  # and the buffers will match

Available Transforms
--------------------

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    Transform
    TransformedEnv
    ActionDiscretizer
    ActionMask
    AutoResetEnv
    AutoResetTransform
    BatchSizeTransform
    BinarizeReward
    BurnInTransform
    CatFrames
    CatTensors
    CenterCrop
    ClipTransform
    Compose
    ConditionalPolicySwitch
    ConditionalSkip
    Crop
    DTypeCastTransform
    DeviceCastTransform
    DiscreteActionProjection
    DoubleToFloat
    EndOfLifeTransform
    ExcludeTransform
    FiniteTensorDictCheck
    FlattenObservation
    FrameSkipTransform
    GrayScale
    Hash
    InitTracker
    KLRewardTransform
    LineariseRewards
    ModuleTransform
    MultiAction
    NoopResetEnv
    ObservationNorm
    ObservationTransform
    PermuteTransform
    PinMemoryTransform
    R3MTransform
    RandomCropTensorDict
    RemoveEmptySpecs
    RenameTransform
    Resize
    Reward2GoTransform
    RewardClipping
    RewardScaling
    RewardSum
    SelectTransform
    SignTransform
    SqueezeTransform
    Stack
    StepCounter
    TargetReturn
    TensorDictPrimer
    TimeMaxPool
    Timer
    Tokenizer
    ToTensorImage
    TrajCounter
    UnaryTransform
    UnsqueezeTransform
    VC1Transform
    VIPRewardTransform
    VIPTransform
    VecGymEnvTransform
    VecNorm
    VecNormV2
    gSDENoise

Environments with masked actions
--------------------------------

In some environments with discrete actions, the actions available to the agent might change throughout execution.
In such cases the environments will output an action mask (under the ``"action_mask"`` key by default).
This mask needs to be used to filter out unavailable actions for that step.

If you are using a custom policy you can pass this mask to your probability distribution like so:

.. code-block::
   :caption: Categorical policy with action mask

        >>> from tensordict.nn import TensorDictModule, ProbabilisticTensorDictModule, TensorDictSequential
        >>> import torch.nn as nn
        >>> from torchrl.modules import MaskedCategorical
        >>> module = TensorDictModule(
        >>>     nn.Linear(in_feats, out_feats),
        >>>     in_keys=["observation"],
        >>>     out_keys=["logits"],
        >>> )
        >>> dist = ProbabilisticTensorDictModule(
        >>>     in_keys={"logits": "logits", "mask": "action_mask"},
        >>>     out_keys=["action"],
        >>>     distribution_class=MaskedCategorical,
        >>> )
        >>> actor = TensorDictSequential(module, dist)

If you want to use a default policy, you will need to wrap your environment in the :class:`~torchrl.envs.transforms.ActionMask`
transform. This transform can take care of updating the action mask in the action spec in order for the default policy
to always know what the latest available actions are. You can do this like so:

.. code-block::
   :caption: How to use the action mask transform

        >>> from tensordict.nn import TensorDictModule, ProbabilisticTensorDictModule, TensorDictSequential
        >>> import torch.nn as nn
        >>> from torchrl.envs.transforms import TransformedEnv, ActionMask
        >>> env = TransformedEnv(
        >>>     your_base_env
        >>>     ActionMask(action_key="action", mask_key="action_mask"),
        >>> )

.. note::
  In case you are using a parallel environment it is important to add the transform to the parallel environment itself
  and not to its sub-environments.
