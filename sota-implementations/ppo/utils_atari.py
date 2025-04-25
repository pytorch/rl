# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch.nn
import torch.optim
from tensordict.nn import TensorDictModule
from torchrl.data.tensor_specs import CategoricalBox
from torchrl.envs import (
    CatFrames,
    DoubleToFloat,
    EndOfLifeTransform,
    EnvCreator,
    ExplorationType,
    GrayScale,
    GymEnv,
    NoopResetEnv,
    ParallelEnv,
    RenameTransform,
    Resize,
    RewardSum,
    set_gym_backend,
    SignTransform,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
    VecNorm,
)
from torchrl.modules import (
    ActorValueOperator,
    ConvNet,
    MLP,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.record import VideoRecorder


# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def make_base_env(
    env_name="BreakoutNoFrameskip-v4",
    frame_skip=4,
    gym_backend="gymnasium",
    is_test=False,
):
    with set_gym_backend(gym_backend):
        env = GymEnv(
            env_name,
            frame_skip=frame_skip,
            from_pixels=True,
            pixels_only=False,
            device="cpu",
            categorical_action_encoding=True,
        )
    env = TransformedEnv(env)
    env.append_transform(NoopResetEnv(noops=30, random=True))
    if not is_test:
        env.append_transform(EndOfLifeTransform())
    return env


def make_parallel_env(env_name, num_envs, device, gym_backend, is_test=False):
    env = ParallelEnv(
        num_envs,
        EnvCreator(lambda: make_base_env(env_name, gym_backend=gym_backend)),
        serial_for_single=True,
        device=device,
    )
    env = TransformedEnv(env)
    env.append_transform(RenameTransform(in_keys=["pixels"], out_keys=["pixels_int"]))
    env.append_transform(ToTensorImage(in_keys=["pixels_int"], out_keys=["pixels"]))
    env.append_transform(GrayScale())
    env.append_transform(Resize(84, 84))
    env.append_transform(CatFrames(N=4, dim=-3))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter(max_steps=4500))
    if not is_test:
        env.append_transform(SignTransform(in_keys=["reward"]))
    env.append_transform(DoubleToFloat())
    env.append_transform(VecNorm(in_keys=["pixels"]))
    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_ppo_modules_pixels(proof_environment, device):

    # Define input shape
    input_shape = proof_environment.observation_spec["pixels"].shape

    # Define distribution class and kwargs
    if isinstance(proof_environment.action_spec_unbatched.space, CategoricalBox):
        num_outputs = proof_environment.action_spec_unbatched.space.n
        distribution_class = torch.distributions.Categorical
        distribution_kwargs = {}
    else:  # is ContinuousBox
        num_outputs = proof_environment.action_spec_unbatched.shape
        distribution_class = TanhNormal
        distribution_kwargs = {
            "low": proof_environment.action_spec_unbatched.space.low.to(device),
            "high": proof_environment.action_spec_unbatched.space.high.to(device),
        }

    # Define input keys
    in_keys = ["pixels"]

    # Define a shared Module and TensorDictModule (CNN + MLP)
    common_cnn = ConvNet(
        activation_class=torch.nn.ReLU,
        num_cells=[32, 64, 64],
        kernel_sizes=[8, 4, 3],
        strides=[4, 2, 1],
        device=device,
    )
    common_cnn_output = common_cnn(torch.ones(input_shape, device=device))
    common_mlp = MLP(
        in_features=common_cnn_output.shape[-1],
        activation_class=torch.nn.ReLU,
        activate_last_layer=True,
        out_features=512,
        num_cells=[],
        device=device,
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
        device=device,
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
        spec=proof_environment.full_action_spec_unbatched.to(device),
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
        device=device,
    )
    value_module = ValueOperator(
        value_net,
        in_keys=["common_features"],
    )

    return common_module, policy_module, value_module


def make_ppo_models(env_name, device, gym_backend):

    proof_environment = make_parallel_env(
        env_name, 1, device=device, gym_backend=gym_backend
    )
    common_module, policy_module, value_module = make_ppo_modules_pixels(
        proof_environment,
        device=device,
    )

    # Wrap modules in a single ActorCritic operator
    actor_critic = ActorValueOperator(
        common_operator=common_module,
        policy_operator=policy_module,
        value_operator=value_module,
    )

    with torch.no_grad():
        td = proof_environment.fake_tensordict().expand(10)
        actor_critic(td)
        del td

    actor = actor_critic.get_policy_operator()
    critic = actor_critic.get_value_operator()

    del proof_environment

    return actor, critic


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()


def eval_model(actor, test_env, num_episodes=3):
    test_rewards = []
    for _ in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        test_env.apply(dump_video)
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards.append(reward.cpu())
    del td_test
    return torch.cat(test_rewards, 0).mean()
