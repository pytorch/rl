# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
This code exemplifies how an actor that uses a RNN backbone can be built.

It is based on snippets from the DQN with RNN tutorial.

There are two main APIs to be aware of when using RNNs, and dedicated notes regarding these can be found at the end
of this example: the `set_recurrent_mode` context manager, and the `make_tensordict_primer` method.

"""
from collections import OrderedDict

import torch
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
from torch import nn

from torchrl.envs import (
    Compose,
    GrayScale,
    GymEnv,
    InitTracker,
    ObservationNorm,
    Resize,
    RewardScaling,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.modules import ConvNet, LSTMModule, MLP, QValueModule, set_recurrent_mode

# Define the device to use for computations (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create a transformed environment using the CartPole-v1 gym environment
env = TransformedEnv(
    GymEnv("CartPole-v1", from_pixels=True, device=device),
    # Apply a series of transformations to the environment:
    # 1. Convert observations to tensor images
    # 2. Convert images to grayscale
    # 3. Resize images to 84x84 pixels
    # 4. Keep track of the step count
    # 5. Initialize a tracker for the environment
    # 6. Scale rewards by a factor of 0.1
    # 7. Normalize observations to have zero mean and unit variance (we'll adapt that dynamically later)
    Compose(
        ToTensorImage(),
        GrayScale(),
        Resize(84, 84),
        StepCounter(),
        InitTracker(),
        RewardScaling(loc=0.0, scale=0.1),
        ObservationNorm(standard_normal=True, in_keys=["pixels"]),
    ),
)

# Initialize the normalization statistics for the observation norm transform
env.transform[-1].init_stats(1000, reduce_dim=[0, 1, 2], cat_dim=0, keep_dims=[0])

# Reset the environment to get an initial observation
td = env.reset()

# Define a feature extractor module that takes pixel observations as input
# and outputs an embedding vector
feature = Mod(
    ConvNet(
        num_cells=[32, 32, 64],
        squeeze_output=True,
        aggregator_class=nn.AdaptiveAvgPool2d,
        aggregator_kwargs={"output_size": (1, 1)},
        device=device,
    ),
    in_keys=["pixels"],
    out_keys=["embed"],
)

# Get the shape of the embedding vector output by the feature extractor
with torch.no_grad():
    n_cells = feature(env.reset())["embed"].shape[-1]

# Define an LSTM module that takes the embedding vector as input and outputs
# a new embedding vector
lstm = LSTMModule(
    input_size=n_cells,
    hidden_size=128,
    device=device,
    in_key="embed",
    out_key="embed",
)

# Define a multi-layer perceptron (MLP) module that takes the LSTM output as
# input and outputs action values
mlp = MLP(
    out_features=2,
    num_cells=[
        64,
    ],
    device=device,
)

# Initialize the bias of the last layer of the MLP to zero
mlp[-1].bias.data.fill_(0.0)

# Wrap the MLP in a TensorDictModule to handle input/output keys
mlp = Mod(mlp, in_keys=["embed"], out_keys=["action_value"])

# Define a Q-value module that computes the Q-value of the current state
qval = QValueModule(action_space=None, spec=env.action_spec)

# Add a TensorDictPrimer to the environment to ensure that the policy is aware
# of the supplementary inputs and outputs (recurrent states) during rollout execution
# This is necessary when using batched environments or parallel data collection
env.append_transform(lstm.make_tensordict_primer())

# Create a sequential module that combines the feature extractor, LSTM, MLP, and Q-value modules
policy = Seq(OrderedDict(feature=feature, lstm=lstm, mlp=mlp, qval=qval))

# Roll out the policy in the environment for 100 steps
rollout = env.rollout(100, policy)
print(rollout)

# Print result:
#
# TensorDict(
#     fields={
#         action: Tensor(shape=torch.Size([10, 2]), device=cpu, dtype=torch.int64, is_shared=False),
#         action_value: Tensor(shape=torch.Size([10, 2]), device=cpu, dtype=torch.float32, is_shared=False),
#         chosen_action_value: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
#         done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
#         embed: Tensor(shape=torch.Size([10, 128]), device=cpu, dtype=torch.float32, is_shared=False),
#         is_init: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
#         next: TensorDict(
#             fields={
#                 done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
#                 is_init: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
#                 pixels: Tensor(shape=torch.Size([10, 1, 84, 84]), device=cpu, dtype=torch.float32, is_shared=False),
#                 recurrent_state_c: Tensor(shape=torch.Size([10, 1, 128]), device=cpu, dtype=torch.float32, is_shared=False),
#                 recurrent_state_h: Tensor(shape=torch.Size([10, 1, 128]), device=cpu, dtype=torch.float32, is_shared=False),
#                 reward: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
#                 step_count: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.int64, is_shared=False),
#                 terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
#                 truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
#             batch_size=torch.Size([10]),
#             device=cpu,
#             is_shared=False),
#         pixels: Tensor(shape=torch.Size([10, 1, 84, 84]), device=cpu, dtype=torch.float32, is_shared=False),
#         recurrent_state_c: Tensor(shape=torch.Size([10, 1, 128]), device=cpu, dtype=torch.float32, is_shared=False),
#         recurrent_state_h: Tensor(shape=torch.Size([10, 1, 128]), device=cpu, dtype=torch.float32, is_shared=False),
#         step_count: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.int64, is_shared=False),
#         terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
#         truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
#     batch_size=torch.Size([10]),
#     device=cpu,
#     is_shared=False)
#

# Notes:
# 1. make_tensordict_primer
#
# Regarding make_tensordict_primer, it creates a TensorDictPrimer object that ensures the policy is aware
# of the supplementary inputs and outputs (recurrent states) during rollout execution.
# This is necessary when using batched environments or parallel data collection, as the recurrent states
# need to be shared across processes and dealt with properly.
#
# In other words, make_tensordict_primer adds the LSTM's hidden states to the environment's specs,
# allowing the environment to properly handle the recurrent states during rollouts. Without it, the policy
# would not be able to use the LSTM's memory buffers correctly, leading to poorly defined behaviors,
# especially in parallel settings.
#
# By adding the TensorDictPrimer to the environment, you ensure that the policy can correctly use the
# LSTM's recurrent states, even when running in parallel or batched environments. This is why
# env.append_transform(lstm.make_tensordict_primer()) is called before creating the policy and rolling it
# out in the environment.
#
# 2. Using the LSTM to process multiple steps at once.
#
# When set_recurrent_mode("recurrent") is used, the LSTM will process the entire input tensordict as a sequence, using
# its recurrent connections to maintain state across time steps. This mode may utilize CuDNN to accelerate the processing
# of the sequence on CUDA devices. The behavior in this mode is akin to torch.nn.LSTM, where the LSTM expects the input
# data to be organized in batches of sequences.
#
# On the other hand, when set_recurrent_mode("sequential") is used, the
# LSTM will process each step in the input tensordict independently, without maintaining any state across time steps. This
# mode makes the LSTM behave similarly to torch.nn.LSTMCell, where each input is treated as a separate, independent
# element.
#
# In the example code, set_recurrent_mode("recurrent") is used to process a tensordict of shape [T], where T
# is the number of steps. This allows the LSTM to use its recurrent connections to maintain state across the entire
# sequence.
#
# In contrast, set_recurrent_mode("sequential") is used to process a single step from the tensordict (i.e.,
# rollout[0]). In this case, the LSTM does not use its recurrent connections, and simply processes the single step as if
# it were an independent input.

with set_recurrent_mode("recurrent"):
    # Process a tensordict of shape [T] where T is a number of steps
    print(policy(rollout))

with set_recurrent_mode("sequential"):
    # Process a tensordict of shape [T] where T is a number of steps
    print(policy(rollout[0]))
