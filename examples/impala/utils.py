# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch.nn
import torch.optim
from tensordict.nn import TensorDictModule
from torchrl.data import CompositeSpec, UnboundedDiscreteTensorSpec
from torchrl.envs import (
    CatFrames,
    DoubleToFloat,
    ExplorationType,
    GrayScale,
    GymEnv,
    NoopResetEnv,
    Resize,
    RewardClipping,
    RewardSum,
    StepCounter,
    ToTensorImage,
    Transform,
    TransformedEnv,
    VecNorm,
)
from torchrl.modules import (
    ActorValueOperator,
    ConvNet,
    MLP,
    OneHotCategorical,
    ProbabilisticActor,
    ValueOperator,
)


# To pickle the environment, in particular the EndOfLifeTransform, we need to
# add the utils path to the PYTHONPATH
utils_path = os.path.abspath(os.path.abspath(os.path.dirname(__file__)))
current_pythonpath = os.environ.get("PYTHONPATH", "")
new_pythonpath = f"{utils_path}:{current_pythonpath}"
os.environ["PYTHONPATH"] = new_pythonpath


# ====================================================================
# Environment utils
# --------------------------------------------------------------------


class EndOfLifeTransform(Transform):
    def _step(self, tensordict, next_tensordict):
        # lives = self.parent.base_env._env.unwrapped.ale.lives()
        lives = 0
        end_of_life = torch.tensor(
            [tensordict["lives"] < lives], device=self.parent.device
        )
        end_of_life = end_of_life | next_tensordict.get("done")
        next_tensordict.set("eol", end_of_life)
        next_tensordict.set("lives", lives)
        return next_tensordict

    def reset(self, tensordict):
        lives = self.parent.base_env._env.unwrapped.ale.lives()
        end_of_life = False
        tensordict.set("eol", [end_of_life])
        tensordict.set("lives", lives)
        return tensordict

    def transform_observation_spec(self, observation_spec):
        full_done_spec = self.parent.output_spec["full_done_spec"]
        observation_spec["eol"] = full_done_spec["done"].clone()
        observation_spec["lives"] = UnboundedDiscreteTensorSpec(
            self.parent.batch_size, device=self.parent.device
        )
        return observation_spec


def make_env(env_name, device, is_test=False):
    env = GymEnv(
        env_name, frame_skip=4, from_pixels=True, pixels_only=False, device=device
    )
    env = TransformedEnv(env)
    env.append_transform(NoopResetEnv(noops=30, random=True))
    if not is_test:
        env.append_transform(EndOfLifeTransform())
        env.append_transform(RewardClipping(-1, 1))
    env.append_transform(ToTensorImage(from_int=False))
    env.append_transform(GrayScale())
    env.append_transform(Resize(84, 84))
    env.append_transform(CatFrames(N=4, dim=-3))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter(max_steps=4500))
    env.append_transform(DoubleToFloat())
    env.append_transform(VecNorm(in_keys=["pixels"]))
    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_ppo_modules_pixels(proof_environment):

    # Define input shape
    input_shape = proof_environment.observation_spec["pixels"].shape

    # Define distribution class and kwargs
    num_outputs = proof_environment.action_spec.space.n
    distribution_class = OneHotCategorical
    distribution_kwargs = {}

    # Define input keys
    in_keys = ["pixels"]

    # Define a shared Module and TensorDictModule (CNN + MLP)
    common_cnn = ConvNet(
        activation_class=torch.nn.ReLU,
        num_cells=[32, 64, 64],
        kernel_sizes=[8, 4, 3],
        strides=[4, 2, 1],
    )
    common_cnn_output = common_cnn(torch.ones(input_shape))
    common_mlp = MLP(
        in_features=common_cnn_output.shape[-1],
        activation_class=torch.nn.ReLU,
        activate_last_layer=True,
        out_features=512,
        num_cells=[],
    )
    common_mlp_output = common_mlp(common_cnn_output)

    # Define shared net as TensorDictModule
    common_module = TensorDictModule(
        module=torch.nn.Sequential(common_cnn, common_mlp),
        in_keys=in_keys,
        out_keys=["common_features"],
    )

    # Define on head for the policy
    policy_net = MLP(
        in_features=common_mlp_output.shape[-1],
        out_features=num_outputs,
        activation_class=torch.nn.ReLU,
        num_cells=[],
    )
    policy_module = TensorDictModule(
        module=policy_net,
        in_keys=["common_features"],
        out_keys=["logits"],
    )

    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        policy_module,
        in_keys=["logits"],
        spec=CompositeSpec(action=proof_environment.action_spec),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Define another head for the value
    value_net = MLP(
        activation_class=torch.nn.ReLU,
        in_features=common_mlp_output.shape[-1],
        out_features=1,
        num_cells=[],
    )
    value_module = ValueOperator(
        value_net,
        in_keys=["common_features"],
    )

    return common_module, policy_module, value_module


def make_ppo_models(env_name):

    proof_environment = make_env(env_name, device="cpu")
    common_module, policy_module, value_module = make_ppo_modules_pixels(
        proof_environment
    )

    # Wrap modules in a single ActorCritic operator
    actor_critic = ActorValueOperator(
        common_operator=common_module,
        policy_operator=policy_module,
        value_operator=value_module,
    )

    actor = actor_critic.get_policy_operator()
    critic = actor_critic.get_value_operator()

    del proof_environment

    return actor, critic


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def eval_model(actor, test_env, num_episodes=3):
    test_rewards = torch.zeros(num_episodes, dtype=torch.float32)
    for i in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards[i] = reward.sum()
    del td_test
    return test_rewards.mean()
