"""
Exporting TorchRL modules
=========================

**Author**: `Vincent Moens <https://github.com/vmoens>`_

.. _export_tuto:

.. note:: To run this tutorial in a notebook, add an installation cell
  at the beginning containing:

    .. code-block::

        !pip install tensordict
        !pip install torchrl
        !pip install "gymnasium[atari]"

Introduction
------------

Learning a policy has little value if that policy cannot be deployed in real-world settings.
As shown in other tutorials, TorchRL has a strong focus on modularity and composability: thanks to ``tensordict``,
the components of the library can be written in the most generic way there is by abstracting their signature to a
mere set of operations on an input ``TensorDict``.
This may give the impression that the library is bound to be used only for training, as typical low-level execution
hardwares (edge devices, robots, arduino, Raspberry Pi) do not execute python code, let alone with pytorch, tensordict
or torchrl installed.

Fortunately, PyTorch provides a full ecosystem of solutions to export code and trained models to devices and
hardwares, and TorchRL is fully equipped to interact with it.
It is possible to choose from a varied set of backends, including ONNX or AOTInductor examplified in this tutorial.
This tutorial gives a quick overview of how a trained model can be isolated and shipped as a standalone executable
to be exported on hardware.

Key learnings:

- Export any TorchRL module after training;
- Using various backends;
- Testing your exported model.

Fast recap: a simple TorchRL training loop
------------------------------------------

In this section, we reproduce the training loop from the last Getting Started tutorial, slightly adapted to be used
with Atari games as they are rendered by the gymnasium library.
We will stick to the DQN example, and show how a policy that outputs a distribution over values can be used instead
later.

"""
import time
from pathlib import Path

import numpy as np

import torch

from tensordict.nn import (
    TensorDictModule as Mod,
    TensorDictSequential,
    TensorDictSequential as Seq,
)

from torch.optim import Adam

from torchrl._utils import timeit
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer

from torchrl.envs import (
    Compose,
    GrayScale,
    GymEnv,
    Resize,
    set_exploration_type,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
)

from torchrl.modules import ConvNet, EGreedyModule, QValueModule

from torchrl.objectives import DQNLoss, SoftUpdate

torch.manual_seed(0)

env = TransformedEnv(
    GymEnv("ALE/Pong-v5", categorical_action_encoding=True),
    Compose(
        ToTensorImage(), Resize(84, interpolation="nearest"), GrayScale(), StepCounter()
    ),
)
env.set_seed(0)

value_mlp = ConvNet.default_atari_dqn(num_actions=env.action_spec.space.n)
value_net = Mod(value_mlp, in_keys=["pixels"], out_keys=["action_value"])
policy = Seq(value_net, QValueModule(spec=env.action_spec))
exploration_module = EGreedyModule(
    env.action_spec, annealing_num_steps=100_000, eps_init=0.5
)
policy_explore = Seq(policy, exploration_module)

init_rand_steps = 5000
frames_per_batch = 100
optim_steps = 10
collector = SyncDataCollector(
    env,
    policy_explore,
    frames_per_batch=frames_per_batch,
    total_frames=-1,
    init_random_frames=init_rand_steps,
)
rb = ReplayBuffer(storage=LazyTensorStorage(100_000))

loss = DQNLoss(value_network=policy, action_space=env.action_spec, delay_value=True)
optim = Adam(loss.parameters())
updater = SoftUpdate(loss, eps=0.99)

total_count = 0
total_episodes = 0
t0 = time.time()
for data in collector:
    # Write data in replay buffer
    rb.extend(data)
    max_length = rb[:]["next", "step_count"].max()
    if len(rb) > init_rand_steps:
        # Optim loop (we do several optim steps
        # per batch collected for efficiency)
        for _ in range(optim_steps):
            sample = rb.sample(128)
            loss_vals = loss(sample)
            loss_vals["loss"].backward()
            optim.step()
            optim.zero_grad()
            # Update exploration factor
            exploration_module.step(data.numel())
            # Update target params
            updater.step()
            total_count += data.numel()
            total_episodes += data["next", "done"].sum()
    if max_length > 200:
        break

#####################################
# Exporting a TensorDictModule-based policy
# -----------------------------------------
#
# ``TensorDict`` allowed us to build a policy with a great flexibility: from a regular :class:`~torch.nn.Module` that
# outputs action values from an observation, we added a :class:`~torchrl.modules.QValueModule` module that
# read these values and computed an action using some heuristic (e.g., an argmax call).
#
# However, there's a small technical catch in our case: the environment (the actual Atari game) doesn't return
# grayscale, 84x84 images but raw screen-size color ones. The transforms we appended to the environment make sure that
# the images can be read by the model. We can see that, from the training perspective, the boundary between environment
# and model is blurry, but at execution time things are much clearer: the model should take care of transforming
# the input data (images) to the format that can be processed by our CNN.
#
# Here again, the magic of tensordict will unblock us: it happens that most of local (non-recursive) TorchRL's
# transforms can be used both as environment transforms or preprocessing blocks within a :class:`~torch.nn.Module`
# instance. Let's see how we can prepend them to our policy:

policy_transform = TensorDictSequential(
    env.transform[
        :-1
    ],  # the last transform is a step counter which we don't need for preproc
    policy_explore.requires_grad_(
        False
    ),  # Using the explorative version of the policy for didactic purposes, see below.
)
#####################################
# We create a fake input, and pass it to :func:`~torch.export.export` with the policy. This will give a "raw" python
# function that will read our input tensor and output an action without any reference to TorchRL or tensordict modules.
#
# A good practice is to call :meth:`~tensordict.nn.TensorDictSequential.select_out_keys` to let the model know that
# we only want a certain set of outputs (in case the policy returns more than one tensor).
#

fake_td = env.base_env.fake_tensordict()
pixels = fake_td["pixels"]
with set_exploration_type("DETERMINISTIC"):
    exported_policy = torch.export.export(
        # Select only the "action" output key
        policy_transform.select_out_keys("action"),
        args=(),
        kwargs={"pixels": pixels},
        strict=False,
    )

#####################################
# Representing the policy can be quite insightful: we can see that the first operations are a permute, a div, unsqueeze,
# resize followed by the convolutional and MLP layers.
#
print("Deterministic policy")
exported_policy.graph_module.print_readable()

#####################################
# As a final check, we can execute the policy with a dummy input. The output (for a single image) should be an integer
# from 0 to 6 representing the action to be executed in the game.

output = exported_policy.module()(pixels=pixels)
print("Exported module output", output)

#####################################
# Further details on exporting :class:`~tensordict.nn.TensorDictModule` instances can be found in the tensordict
# `documentation <https://pytorch.org/tensordict/stable/tutorials/export.html>`_.
#
# .. note::
#    Exporting modules that take and output nested keys is perfectly fine.
#    The corresponding kwargs will be the `"_".join(key)` version of the key, i.e., the `("group0", "agent0", "obs")`
#    key will correspond to the `"group0_agent0_obs"` keyword argument. Colliding keys (e.g., `("group0_agent0", "obs")`
#    and `("group0", "agent0_obs")` may lead to undefined behaviours and should be avoided at all cost.
#    Obviously, key names should also always produce valid keyword arguments, i.e., they should not contain special
#    characters such as spaces or commas.
#
# ``torch.export`` has many other features that we will explore further below. Before this, let us just do a small
# digression on exploration and stochastic policies in the context of test-time inference, as well as recurrent
# policies.
#
# Working with stochastic policies
# --------------------------------
#
# As you probably noted, above we used the :class:`~torchrl.envs.set_exploration_type` context manager to control
# the behaviour of the policy. If the policy is stochastic (e.g., the policy outputs a distribution over the action
# space like it is the case in PPO or other similar on-policy algorithms) or explorative (with an exploration module
# appended like E-Greedy, additive gaussian or Ornstein-Uhlenbeck) we may want or not want to use that exploration
# strategy in its exported version.
# Fortunately, export utils can understand that context manager and as long as the exportation occurs within the right
# context manager, the behaviour of the policy should match what is indicated. To demonstrate this, let us try with
# another exploration type:

with set_exploration_type("RANDOM"):
    exported_stochastic_policy = torch.export.export(
        policy_transform.select_out_keys("action"),
        args=(),
        kwargs={"pixels": pixels},
        strict=False,
    )

#####################################
# Our exported policy should now have a random module at the end of the call stack, unlike the previous version.
# Indeed, the last three operations are: generate a random integer between 0 and 6, use a random mask and select
# the network output or the random action based on the value in the mask.
#
print("Stochastic policy")
exported_stochastic_policy.graph_module.print_readable()

#####################################
# Working with recurrent policies
# -------------------------------
#
# Another typical use case is a recurrent policy that will output an action as well as a one or more recurrent state.
# LSTM and GRU are CuDNN-based modules, which means that they will behave differently than regular
# :class:`~torch.nn.Module` instances (export utils may not trace them well). Fortunately, TorchRL provides a python
# implementation of these modules that can be swapped with the CuDNN version when desired.
#
# To show this, let us write a prototypical policy that relies on an RNN:
#
from tensordict.nn import TensorDictModule
from torchrl.envs import BatchSizeTransform
from torchrl.modules import LSTMModule, MLP

lstm = LSTMModule(
    input_size=32,
    num_layers=2,
    hidden_size=256,
    in_keys=["observation", "hidden0", "hidden1"],
    out_keys=["intermediate", "hidden0", "hidden1"],
)

#####################################
# If the LSTM module is not python based but CuDNN (:class:`~torch.nn.LSTM`), the :meth:`~torchrl.modules.LSTMModule.make_python_based`
# method can be used to use the python version.
#
lstm = lstm.make_python_based()

#####################################
# Let's now create the policy. We combine two layers that modify the shape of the input (unsqueeze/squeeze operations)
# with the LSTM and an MLP.
#

recurrent_policy = TensorDictSequential(
    # Unsqueeze the first dim of all tensors to make LSTMCell happy
    BatchSizeTransform(reshape_fn=lambda x: x.unsqueeze(0)),
    lstm,
    TensorDictModule(
        MLP(in_features=256, out_features=5, num_cells=[64, 64]),
        in_keys=["intermediate"],
        out_keys=["action"],
    ),
    # Squeeze the first dim of all tensors to get the original shape back
    BatchSizeTransform(reshape_fn=lambda x: x.squeeze(0)),
)

#####################################
# As before, we select the relevant keys:
#

recurrent_policy.select_out_keys("action", "hidden0", "hidden1")
print("recurrent policy input keys:", recurrent_policy.in_keys)
print("recurrent policy output keys:", recurrent_policy.out_keys)

#####################################
# We are now ready to export. To do this, we build fake inputs and pass them to :func:`~torch.export.export`:
#

fake_obs = torch.randn(32)
fake_hidden0 = torch.randn(2, 256)
fake_hidden1 = torch.randn(2, 256)

# Tensor indicating whether the state is the first of a sequence
fake_is_init = torch.zeros((), dtype=torch.bool)

exported_recurrent_policy = torch.export.export(
    recurrent_policy,
    args=(),
    kwargs={
        "observation": fake_obs,
        "hidden0": fake_hidden0,
        "hidden1": fake_hidden1,
        "is_init": fake_is_init,
    },
    strict=False,
)
print("Recurrent policy graph:")
exported_recurrent_policy.graph_module.print_readable()

#####################################
# AOTInductor: Export your policy to pytorch-free C++ binaries
# ------------------------------------------------------------
#
# AOTInductor is a PyTorch module that allows you to export your model (policy or other) to pytorch-free C++ binaries.
# This is particularly useful when you need to deploy your model on devices or platforms where PyTorch is not available.
#
# Here's an example of how you can use AOTInductor to export your policy, inspired by the
# `AOTI documentation <https://pytorch.org/docs/main/torch.compiler_aot_inductor.html>`_:
#

from tempfile import TemporaryDirectory

from torch._inductor import aoti_compile_and_package, aoti_load_package

with TemporaryDirectory() as tmpdir:
    path = str(Path(tmpdir) / "model.pt2")
    with torch.no_grad():
        pkg_path = aoti_compile_and_package(
            exported_policy,
            # Specify the generated shared library path
            package_path=path,
        )
    print("pkg_path", pkg_path)

    compiled_module = aoti_load_package(pkg_path)

print(compiled_module(pixels=pixels))

#####################################
#
# Exporting TorchRL models with ONNX
# ----------------------------------
#
# .. note:: To execute this part of the script, make sure pytorch onnx is installed:
#
#
#     .. code-block::
#
#         !pip install onnx-pytorch
#         !pip install onnxruntime
#
# You can also find more information about using ONNX in the PyTorch ecosystem
# `here <https://pytorch.org/tutorials/beginner/onnx/intro_onnx.html>`_. The following example is based on this
# documentation.
#
# In this section, we are going to showcase how we can export our model in such a way that it can be
# executed on a pytorch-free setting.
#
# There are plenty of resources on the web explaining how ONNX can be used to deploy PyTorch models on various
# hardwares and devices, including `Raspberry Pi <https://qengineering.eu/install-pytorch-on-raspberry-pi-4.html>`_,
# `NVIDIA TensorRT <https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html>`_,
# `iOS <https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html>`_ and
# `Android <https://onnxruntime.ai/docs/tutorials/mobile/>`_.
#
# The Atari game we trained on can be isolated without TorchRL or gymnasium with the
# `ALE library <https://github.com/Farama-Foundation/Arcade-Learning-Environment>`_ and therefore provides us with
# a good example of what we can achieve with ONNX.
#
# Let us see what this API looks like:

from ale_py import ALEInterface, roms

# Create the interface
ale = ALEInterface()
# Load the pong environment
ale.loadROM(roms.Pong)
ale.reset_game()

# Make a step in the simulator
action = 0
reward = ale.act(action)
screen_obs = ale.getScreenRGB()
print("Observation from ALE simulator:", type(screen_obs), screen_obs.shape)

from matplotlib import pyplot as plt

plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.imshow(screen_obs)
plt.title("Screen rendering of Pong game.")

#####################################
# Exporting to ONNX is quite similar the Export/AOTI above:
#

import onnxruntime

with set_exploration_type("DETERMINISTIC"):
    # We use torch.onnx.dynamo_export to capture the computation graph from our policy_explore model
    pixels = torch.as_tensor(screen_obs)
    onnx_policy_export = torch.onnx.dynamo_export(policy_transform, pixels=pixels)

#####################################
# We can now save the program on disk and load it:
with TemporaryDirectory() as tmpdir:
    onnx_file_path = str(Path(tmpdir) / "policy.onnx")
    onnx_policy_export.save(onnx_file_path)

    ort_session = onnxruntime.InferenceSession(
        onnx_file_path, providers=["CPUExecutionProvider"]
    )

onnxruntime_input = {ort_session.get_inputs()[0].name: screen_obs}
onnx_policy = ort_session.run(None, onnxruntime_input)

#####################################
# Running a rollout with ONNX
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We now have an ONNX model that runs our policy. Let's compare it to the original TorchRL instance: because it is
# more lightweight, the ONNX version should be faster than the TorchRL one.


def onnx_policy(screen_obs: np.ndarray) -> int:
    onnxruntime_input = {ort_session.get_inputs()[0].name: screen_obs}
    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
    action = int(onnxruntime_outputs[0])
    return action


with timeit("ONNX rollout"):
    num_steps = 1000
    ale.reset_game()
    for _ in range(num_steps):
        screen_obs = ale.getScreenRGB()
        action = onnx_policy(screen_obs)
        reward = ale.act(action)

with timeit("TorchRL version"), torch.no_grad(), set_exploration_type("DETERMINISTIC"):
    env.rollout(num_steps, policy_explore)

print(timeit.print())

#####################################
# Note that ONNX also offers the possibility of optimizing models directly, but this is beyond the scope of this
# tutorial.
#
# Conclusion
# ----------
#
# In this tutorial, we learned how to export TorchRL modules using various backends such as PyTorch's built-in export
# functionality, ``AOTInductor``, and ``ONNX``.
# We demonstrated how to export a policy trained on an Atari game and run it on a pytorch-free setting using the ``ALE``
# library. We also compared the performance of the original TorchRL instance with the exported ONNX model.
#
# Key takeaways:
#
# - Exporting TorchRL modules allows for deployment on devices without PyTorch installed.
# - AOTInductor and ONNX provide alternative backends for exporting models.
# - Optimizing ONNX models can improve performance.
#
# Further reading and learning steps:
#
# - Check out the official documentation for PyTorch's `export functionality <https://pytorch.org/docs/stable/export.html>`_,
#   `AOTInductor <https://pytorch.org/tutorials/recipes/torch_export_aoti_python.html>`_, and
#   `ONNX <https://pytorch.org/tutorials/beginner/onnx/intro_onnx.html>`_ for more
#   information.
# - Experiment with deploying exported models on different devices.
# - Explore optimization techniques for ONNX models to improve performance.
#
