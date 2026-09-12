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
Reusing the example above:

.. code-block::
   :caption: Transform parent

        >>> resize_parent = env.transform[-1].parent  # returns the same as TransformedEnv(base_env, transform[:-1])


Transformed environment can be used with vectorized environments.
Since each transform uses a ``"in_keys"``/``"out_keys"`` set of keyword argument, it is
also easy to root the transform graph to each component of the observation data (e.g.
pixels or states etc).

Forward and inverse transforms
------------------------------

Transforms also have an :meth:`~torchrl.envs.transforms.Transform.inv` method that is called before the action is applied in reverse
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
  :class:`~torchrl.envs.transforms.TransformedEnv`).
- `out_keys` refers to the outside world (outer = `policy`, `agent`, etc.).

For example, with `in_keys=["obs"]` and `out_keys=["obs_standardized"]`, the policy will "see" a standardized
observation, while the base environment outputs a regular observation.

Similarly, for inverse keys:

- `in_keys_inv` refers to entries as seen by the base environment.
- `out_keys_inv` refers to entries as seen or produced by the policy.

The following figure illustrates this concept for the :class:`~torchrl.envs.transforms.RenameTransform` class: the input
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
For example, with :class:`~torchrl.envs.transforms.ActionDiscretizer`, the environment's action (e.g., `"action"`) is a float-valued
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
  :meth:`~torchrl.envs.transforms.Transform._apply_transform` and :meth:`~torchrl.envs.transforms.Transform._inv_apply_transform`.
- The transform needs access to the input data to env.step as well as output? Rewrite
  :meth:`~torchrl.envs.transforms.Transform._step`.
  Otherwise, rewrite :meth:`~torchrl.envs.transforms.Transform._call` (or :meth:`~torchrl.envs.transforms.Transform._inv_call`).
- Is the transform to be used within a replay buffer? Overwrite :meth:`~torchrl.envs.transforms.Transform.forward`,
  :meth:`~torchrl.envs.transforms.Transform.inv`, :meth:`~torchrl.envs.transforms.Transform._apply_transform` or
  :meth:`~torchrl.envs.transforms.Transform._inv_apply_transform`.
- Within a transform, you can access (and make calls to) the parent environment using
  :attr:`~torchrl.envs.transforms.Transform.parent` (the base env + all transforms till this one) or
  :meth:`~torchrl.envs.transforms.Transform.container` (The object that encapsulates the transform).
- Don't forget to edits the specs if needed: top level: :meth:`~torchrl.envs.transforms.Transform.transform_output_spec`,
  :meth:`~torchrl.envs.transforms.Transform.transform_input_spec`.
  Leaf level: :meth:`~torchrl.envs.transforms.Transform.transform_observation_spec`,
  :meth:`~torchrl.envs.transforms.Transform.transform_action_spec`, :meth:`~torchrl.envs.transforms.Transform.transform_state_spec`,
  :meth:`~torchrl.envs.transforms.Transform.transform_reward_spec` and
  :meth:`~torchrl.envs.transforms.Transform.transform_reward_spec`.

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

A transform passed to :class:`~torchrl.data.ReplayBuffer` (the ``transform=``
argument, or :meth:`~torchrl.data.ReplayBuffer.append_transform`) runs on the
**sample** path: ``rb.sample()`` applies it after the storage is indexed.
If the transform implements an inverse, that inverse runs when data is
**written** (``rb.add`` / ``rb.extend``). This is how you store cheap raw
observations and reconstruct a processed view only at training time.

The most common instance of this pattern is frame stacking. The env-side
recipe is below; the collector + replay-buffer recipe (including
``extend`` / ``sample``) lives in
:ref:`Frame stacking images with CatFrames <catframes-collector-replay>`.

.. _catframes-images:

CatFrames for visual RL
~~~~~~~~~~~~~~~~~~~~~~~

:class:`CatFrames` concatenates the last ``N`` observations along one
existing dimension so a feed-forward policy can see motion. There are two
legitimate placements; they solve different problems and must not be
stacked on top of each other.

1. **On the env** (this section). The policy sees a stacked observation
   at every step. The transform is stateful: it keeps a rolling buffer
   that is flushed on reset.
2. **On the replay buffer, at sample time**
   (:ref:`catframes-collector-replay`). Raw unstacked frames are stored;
   the stack is rebuilt when you call ``rb.sample()``. This is the
   memory-efficient path for image off-policy training.

For CHW images (the layout produced by :class:`ToTensorImage`) the stack
dimension is ``dim=-3`` (the channel axis). That is the DQN convention:
a grayscale frame of shape ``[1, H, W]`` becomes ``[N, H, W]``; an RGB
frame of shape ``[3, H, W]`` becomes ``[3 * N, H, W]``. Vector
observations use ``dim=-1`` instead. Using the vector default on pixels,
or stacking along ``-4`` without first inserting that axis, silently
produces the wrong layout.

Env-side stacking (policy input)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Put :class:`CatFrames` on the env so the observation spec the policy
reads is already stacked. Write the stack to a **different** key than
the raw pixels: the collector can then persist the cheap uint8 frame
and drop the float stack (see :ref:`catframes-collector-replay`).

:class:`CatFrames` is stateful. ``env.reset()`` (or a ``"_reset"`` flag
in the tensordict) flushes the rolling buffer and pads the new stack.
With the default ``padding="same"`` the missing history is a repeat of
the first post-reset frame; ``padding="constant"`` fills with
``padding_value`` (0 by default). :class:`InitTracker` is not required
for that flush -- :class:`CatFrames` listens to the env ``_reset`` key
-- but it should sit in the same :class:`Compose` so ``"is_init"`` marks
the steps where the stack was re-initialized. Collectors, recurrent
policies and advantage estimators all read that flag.

.. code-block:: python

    from torchrl.envs import (
        CatFrames,
        Compose,
        GrayScale,
        GymEnv,
        InitTracker,
        Resize,
        StepCounter,
        ToTensorImage,
        TransformedEnv,
    )

    env = TransformedEnv(
        GymEnv("CartPole-v1", from_pixels=True, pixels_only=True),
        Compose(
            InitTracker(),
            ToTensorImage(in_keys=["pixels"], out_keys=["pixels_trsf"]),
            GrayScale(in_keys=["pixels_trsf"]),
            Resize(84, 84, in_keys=["pixels_trsf"]),
            CatFrames(N=4, dim=-3, in_keys=["pixels_trsf"], out_keys=["pixels_trsf"]),
            StepCounter(),
        ),
    )  # doctest: +SKIP
    # "pixels": raw uint8 frame from the env
    # "pixels_trsf": float stack of shape [4, 84, 84] (grayscale, dim=-3)
    # "is_init": True on the first step after every reset

A rollout or a :class:`~torchrl.collectors.Collector` then feeds
``pixels_trsf`` to the policy. Do **not** also attach a second
:class:`CatFrames` on the replay buffer that reads the same stacked
key -- that stacks twice (``[N, H, W]`` stored, ``[N * N, H, W]``
sampled). Either:

- store the already-stacked ``pixels_trsf`` and leave the buffer
  transform-free (``N`` times more RAM), or
- exclude ``pixels_trsf`` on ``extend`` and rebuild the stack from
  raw ``"pixels"`` at sample time (recommended; see
  :ref:`catframes-collector-replay`).

If you want a separate stack axis ``[N, C, H, W]`` rather than channel
concatenation, unsqueeze first and pass ``dim=-4`` (that is the variant
in ``examples/replay-buffers/catframes-in-buffer.py``). For a standard
visual-RL CNN the ``dim=-3`` layout above is the one to use.

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
behaviours. Fortunately, the :class:`Transform` class comes with a :func:`clone`
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
    ActionChunkTransform
    ActionDiscretizer
    ActionMask
    ActionScaling
    ActionTokenizerTransform
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
    DecodeVideoTransform
    DeviceCastTransform
    DiscreteActionProjection
    DoubleToFloat
    EndOfLifeTransform
    ExcludeTransform
    ExpandAs
    FiniteTensorDictCheck
    FlattenAction
    FlattenObservation
    FrameSkipTransform
    GrayScale
    Hash
    HumanoidMacroAction
    InitTracker
    LineariseRewards
    MacroAction
    MacroPrimitive
    MacroPrimitiveTransform
    TargetMacroAction
    CartesianSolver
    RobotMacroAction
    RobotMacroActionMode
    SatelliteMacroAction
    SatelliteAttitudeTransform
    URScriptPrimitive
    MeanActionSelector
    ModuleTransform
    MultiAction
    NextObservationDelta
    NextStateReconstructor
    PolicyAgeFilter
    NoopResetEnv
    ObservationNorm
    ObservationTransform
    PermuteTransform
    PinMemoryTransform
    R3MTransform
    RandomCropTensorDict
    RandomTruncationTransform
    RemoveEmptySpecs
    RenameTransform
    Resize
    RNDTransform
    Reward2GoTransform
    RewardClipping
    RewardScaling
    RewardSum
    RunningMeanStd
    SelectTransform
    SignTransform
    SqueezeTransform
    Stack
    StepCounter
    SuccessReward
    TargetReturn
    TensorDictPrimer
    TerminateTransform
    TimeMaxPool
    Timer
    Tokenizer
    ToTensorImage
    TrajCounter
    URScriptPrimitiveTransform
    UnaryTransform
    UnsqueezeTransform
    VC1Transform
    VIPRewardTransform
    VIPTransform
    VecGymEnvTransform
    VecNorm
    VecNormV2
    gSDENoise

Functional transforms
---------------------

Some transforms expose a pure, stateless functional core (the PyTorch
``torch.nn.functional`` / ``torch.nn.Module`` split) that can be reused directly
on plain tensors, outside the transform machinery. The stateful transform
delegates to the functional so that the two stay equivalent.

.. currentmodule:: torchrl.envs.transforms.functional

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    cat_frames

.. currentmodule:: torchrl.envs.transforms

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
