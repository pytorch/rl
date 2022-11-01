[![pytorch](https://circleci.com/gh/pytorch/rl.svg?style=shield)](https://circleci.com/gh/pytorch/rl)
[![Documentation](https://img.shields.io/badge/Documentation-blue.svg)](https://pytorch.org/rl/)
[![codecov](https://codecov.io/gh/pytorch/rl/branch/main/graph/badge.svg?token=HcpK1ILV6r)](https://codecov.io/gh/pytorch/rl)
[![Twitter Follow](https://img.shields.io/twitter/follow/torchrl1?style=social)](https://twitter.com/torchrl1)
[![Python 3.7, 3.8](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue.svg)](https://www.python.org/downloads/)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/pytorch/rl/blob/main/LICENSE)
<a href="https://pypi.org/project/torchrl"><img src="https://img.shields.io/pypi/v/torchrl" alt="pypi version"></a>
<a href="https://pypi.org/project/torchrl-nightly"><img src="https://img.shields.io/pypi/v/torchrl-nightly?label=nightly" alt="pypi nightly version"></a>
[![Downloads](https://static.pepy.tech/personalized-badge/torchrl?period=total&units=international_system&left_color=blue&right_color=orange&left_text=Downloads)](https://pepy.tech/project/torchrl)
[![Downloads](https://static.pepy.tech/personalized-badge/torchrl-nightly?period=total&units=international_system&left_color=blue&right_color=orange&left_text=Downloads%20(nightly))](https://pepy.tech/project/torchrl-nightly)

# TorchRL

## Disclaimer

This library is not officially released yet and is subject to change.

The features are available before an official release so that users and collaborators can get early access and provide feedback. No guarantee of stability, robustness or backward compatibility is provided.

---

**TorchRL** is an open-source Reinforcement Learning (RL) library for PyTorch.

It provides pytorch and **python-first**, low and high level abstractions for RL that are intended to be **efficient**, **modular**, **documented** and properly **tested**.
The code is aimed at supporting research in RL. Most of it is written in python in a highly modular way, such that researchers can easily swap components, transform them or write new ones with little effort.

This repo attempts to align with the existing pytorch ecosystem libraries in that it has a dataset pillar ([torchrl/envs](torchrl/envs)), [transforms](torchrl/envs/transforms), [models](torchrl/modules), data utilities (e.g. collectors and containers), etc.
TorchRL aims at having as few dependencies as possible (python standard library, numpy and pytorch). Common environment libraries (e.g. OpenAI gym) are only optional.

On the low-level end, torchrl comes with a set of highly re-usable functionals for [cost functions](torchrl/objectives/costs), [returns](torchrl/objectives/returns) and data processing.

TorchRL aims at (1) a high modularity and (2) good runtime performance.

## Features

On the high-level end, TorchRL provides:
- [`TensorDict`](torchrl/data/tensordict/tensordict.py),
a convenient data structure<sup>(1)</sup> to pass data from
one object to another without friction.
`TensorDict` makes it easy to re-use pieces of code across environments, models and
algorithms. For instance, here's how to code a rollout in TorchRL:
    <details>
      <summary>Code</summary>

    ```diff
    - obs, done = env.reset()
    + tensordict = env.reset()
    policy = TensorDictModule(
        model,
        in_keys=["observation_pixels", "observation_vector"],
        out_keys=["action"],
    )
    out = []
    for i in range(n_steps):
    -     action, log_prob = policy(obs)
    -     next_obs, reward, done, info = env.step(action)
    -     out.append((obs, next_obs, action, log_prob, reward, done))
    -     obs = next_obs
    +     tensordict = policy(tensordict)
    +     tensordict = env.step(tensordict)
    +     out.append(tensordict)
    +     tensordict = step_mdp(tensordict)  # renames next_observation_* keys to observation_*
    - obs, next_obs, action, log_prob, reward, done = [torch.stack(vals, 0) for vals in zip(*out)]
    + out = torch.stack(out, 0)  # TensorDict supports multiple tensor operations
    ```
    TensorDict abstracts away the input / output signatures of the modules, env, collectors, replay buffers and losses of the library, allowing its primitives
    to be easily recycled across settings.
    Here's another example of an off-policy training loop in TorchRL (assuming that a data collector, a replay buffer, a loss and an optimizer have been instantiated):

    ```diff
    - for i, (obs, next_obs, action, hidden_state, reward, done) in enumerate(collector):
    + for i, tensordict in enumerate(collector):
    -     replay_buffer.add((obs, next_obs, action, log_prob, reward, done))
    +     replay_buffer.add(tensordict)
        for j in range(num_optim_steps):
    -         obs, next_obs, action, hidden_state, reward, done = replay_buffer.sample(batch_size)
    -         loss = loss_fn(obs, next_obs, action, hidden_state, reward, done)
    +         tensordict = replay_buffer.sample(batch_size)
    +         loss = loss_fn(tensordict)
            loss.backward()
            optim.step()
            optim.zero_grad()
    ```
    Again, this training loop can be re-used across algorithms as it makes a minimal number of assumptions about the structure of the data.

    TensorDict supports multiple tensor operations on its device and shape
    (the shape of TensorDict, or its batch size, is the common arbitrary N first dimensions of all its contained tensors):
    ```python
    # stack and cat
    tensordict = torch.stack(list_of_tensordicts, 0)
    tensordict = torch.cat(list_of_tensordicts, 0)
    # reshape
    tensordict = tensordict.view(-1)
    tensordict = tensordict.permute(0, 2, 1)
    tensordict = tensordict.unsqueeze(-1)
    tensordict = tensordict.squeeze(-1)
    # indexing
    tensordict = tensordict[:2]
    tensordict[:, 2] = sub_tensordict
    # device and memory location
    tensordict.cuda()
    tensordict.to("cuda:1")
    tensordict.share_memory_()
    ```
    </details>

    Check our [TensorDict tutorial](tutorials/tensordict.ipynb) for more information.

- An associated [`TensorDictModule` class](torchrl/modules/tensordict_module/common.py) which is [functorch](https://github.com/pytorch/functorch)-compatible!
    <details>
      <summary>Code</summary>

    ```diff
    transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
    + td_module = TensorDictModule(transformer_model, in_keys=["src", "tgt"], out_keys=["out"])
    src = torch.rand((10, 32, 512))
    tgt = torch.rand((20, 32, 512))
    + tensordict = TensorDict({"src": src, "tgt": tgt}, batch_size=[20, 32])
    - out = transformer_model(src, tgt)
    + td_module(tensordict)
    + out = tensordict["out"]
    ```

    The `TensorDictSequential` class allows to branch sequences of `nn.Module` instances in a highly modular way.
    For instance, here is an implementation of a transformer using the encoder and decoder blocks:
    ```python
    encoder_module = TransformerEncoder(...)
    encoder = TensorDictModule(encoder_module, in_keys=["src", "src_mask"], out_keys=["memory"])
    decoder_module = TransformerDecoder(...)
    decoder = TensorDictModule(decoder_module, in_keys=["tgt", "memory"], out_keys=["output"])
    transformer = TensorDictSequential(encoder, decoder)
    assert transformer.in_keys == ["src", "src_mask", "tgt"]
    assert transformer.out_keys == ["memory", "output"]
    ```

    `TensorDictSequential` allows to isolate subgraphs by querying a set of desired input / output keys:
    ```python
    transformer.select_subsequence(out_keys=["memory"])  # returns the encoder
    transformer.select_subsequence(in_keys=["tgt", "memory"])  # returns the decoder
    ```
    </details>

    The corresponding [tutorial](tutorials/tensordictmodule.ipynb) provides more context about its features.

- a generic [trainer class](torchrl/trainers/trainers.py)<sup>(1)</sup> that
    executes the aforementioned training loop. Through a hooking mechanism,
    it also supports any logging or data transformation operation at any given
    time.

- A common [interface for environments](torchrl/envs)
    which supports common libraries (OpenAI gym, deepmind control lab, etc.)<sup>(1)</sup> and state-less execution (e.g. Model-based environments).
    The [batched environments](torchrl/envs/vec_env.py) containers allow parallel execution<sup>(2)</sup>.
    A common pytorch-first class of [tensor-specification class](torchrl/data/tensor_specs.py) is also provided.
    <details>
      <summary>Code</summary>

    ```python
    env_make = lambda: GymEnv("Pendulum-v1", from_pixels=True)
    env_parallel = ParallelEnv(4, env_make)  # creates 4 envs in parallel
    tensordict = env_parallel.rollout(max_steps=20, policy=None)  # random rollout (no policy given)
    assert tensordict.shape == [4, 20]  # 4 envs, 20 steps rollout
    env_parallel.action_spec.is_in(tensordict["action"])  # spec check returns True
    ```
    </details>

- multiprocess [data collectors](torchrl/collectors/collectors.py)<sup>(2)</sup> that work synchronously or asynchronously.
    Through the use of TensorDict, TorchRL's training loops are made very similar to regular training loops in supervised
    learning (although the "dataloader" -- read data collector -- is modified on-the-fly):
    <details>
      <summary>Code</summary>

    ```python
    env_make = lambda: GymEnv("Pendulum-v1", from_pixels=True)
    collector = MultiaSyncDataCollector(
        [env_make, env_make],
        policy=policy,
        devices=["cuda:0", "cuda:0"],
        total_frames=10000,
        frames_per_batch=50,
        ...
    )
    for i, tensordict_data in enumerate(collector):
        loss = loss_module(tensordict_data)
        loss.backward()
        optim.step()
        optim.zero_grad()
        collector.update_policy_weights_()
    ```
    </details>

- efficient<sup>(2)</sup> and generic<sup>(1)</sup> [replay buffers](torchrl/data/replay_buffers/replay_buffers.py) with modularized storage:
    <details>
      <summary>Code</summary>

    ```python
    storage = LazyMemmapStorage(  # memory-mapped (physical) storage
        cfg.buffer_size,
        scratch_dir="/tmp/"
    )
    buffer = TensorDictPrioritizedReplayBuffer(
        buffer_size=10000,
        alpha=0.7,
        beta=0.5,
        collate_fn=lambda x: x,
        pin_memory=device != torch.device("cpu"),
        prefetch=10,  # multi-threaded sampling
        storage=storage
    )
    ```
    </details>

- cross-library [environment transforms](torchrl/envs/transforms/transforms.py)<sup>(1)</sup>,
    executed on device and in a vectorized fashion<sup>(2)</sup>,
    which process and prepare the data coming out of the environments to be used by the agent:
    <details>
      <summary>Code</summary>

    ```python
    env_make = lambda: GymEnv("Pendulum-v1", from_pixels=True)
    env_base = ParallelEnv(4, env_make, device="cuda:0")  # creates 4 envs in parallel
    env = TransformedEnv(
        env_base,
        Compose(
            ToTensorImage(),
            ObservationNorm(loc=0.5, scale=1.0)),  # executes the transforms once and on device
    )
    tensordict = env.reset()
    assert tensordict.device == torch.device("cuda:0")
    ```
    Other transforms include: reward scaling (`RewardScaling`), shape operations (concatenation of tensors, unsqueezing etc.), contatenation of
    successive operations (`CatFrames`), resizing (`Resize`) and many more.

    Unlike other libraries, the transforms are stacked as a list (and not wrapped in each other), which makes it
    easy to add and remove them at will:
    ```python
    env.insert_transform(0, NoopResetEnv())  # inserts the NoopResetEnv transform at the index 0
    ```
    Nevertheless, transforms can access and execute operations on the parent environment:
    ```python
    transform = env.transform[1]  # gathers the second transform of the list
    parent_env = transform.parent  # returns the base environment of the second transform, i.e. the base env + the first transform
    ```
    </details>

- various tools for distributed learning (e.g. [memory mapped tensors](torchrl/data/tensordict/memmap.py))<sup>(2)</sup>;
- various [architectures](torchrl/modules/models/) and models (e.g. [actor-critic](torchrl/modules/tensordict_module/actors.py))<sup>(1)</sup>:
    <details>
      <summary>Code</summary>

    ```python
    # create an nn.Module
    common_module = ConvNet(
        bias_last_layer=True,
        depth=None,
        num_cells=[32, 64, 64],
        kernel_sizes=[8, 4, 3],
        strides=[4, 2, 1],
    )
    # Wrap it in a TensorDictModule, indicating what key to read in and where to
    # write out the output
    common_module = TensorDictModule(
        common_module,
        in_keys=["pixels"],
        out_keys=["hidden"],
    )
    # Wrap the policy module in NormalParamsWrapper, such that the output
    # tensor is split in loc and scale, and scale is mapped onto a positive space
    policy_module = NormalParamsWrapper(
        MLP(
            num_cells=[64, 64],
            out_features=32,
            activation=nn.ELU,
        )
    )
    # Wrap the nn.Module in a ProbabilisticTensorDictModule, indicating how
    # to build the torch.distribution.Distribution object and what to do with it
    policy_module = ProbabilisticTensorDictModule(  # stochastic policy
        TensorDictModule(
            policy_module,
            in_keys=["hidden"],
            out_keys=["loc", "scale"],
        ),
        dist_in_keys=["loc", "scale"],
        sample_out_key="action",
        distribution_class=TanhNormal,
    )
    value_module = MLP(
        num_cells=[64, 64],
        out_features=1,
        activation=nn.ELU,
    )
    # Wrap the policy and value funciton in a common module
    actor_value = ActorValueOperator(common_module, policy_module, value_module)
    # standalone policy from this
    standalone_policy = actor_value.get_policy_operator()
    ```
    </details>

- exploration [wrappers](torchrl/modules/tensordict_module/exploration.py) and
    [modules](torchrl/modules/models/exploration.py) to easily swap between exploration and exploitation<sup>(1)</sup>:
    <details>
      <summary>Code</summary>

    ```python
    policy_explore = EGreedyWrapper(policy)
    with set_exploration_mode("random"):
        tensordict = policy_explore(tensordict)  # will use eps-greedy
    with set_exploration_mode("mode"):
        tensordict = policy_explore(tensordict)  # will not use eps-greedy
    ```
    </details>

- A series of efficient [loss modules](https://github.com/pytorch/rl/blob/main/torchrl/objectives/costs)
    and highly vectorized
    [functional return and advantage](https://github.com/pytorch/rl/blob/main/torchrl/objectives/returns/functional.py)
    computation.

    <details>
      <summary>Code</summary>

    ### Loss modules
    ```python
    from torchrl.objectives import DQNLoss
    loss_module = DQNLoss(value_network=value_network, gamma=0.99)
    tensordict = replay_buffer.sample(batch_size)
    loss = loss_module(tensordict)
    ```

    ### Advantage computation
    ```python
    from torchrl.objectives.value.functional import vec_td_lambda_return_estimate
    advantage = vec_td_lambda_return_estimate(gamma, lmbda, next_state_value, reward, done)
    ```

    </details>

- various [recipes](torchrl/trainers/helpers/models.py) to build models that
    correspond to the environment being deployed.

If you feel a feature is missing from the library, please submit an issue!
If you would like to contribute to new features, check our [call for contributions](https://github.com/pytorch/rl/issues/509) and our [contribution](CONTRIBUTING.md) page.

## Examples, tutorials and demos

A series of [examples](examples/) are provided with an illustrative purpose:
- [DQN (and add-ons up to Rainbow)](examples/dqn/dqn.py)
- [DDPG](examples/ddpg/ddpg.py)
- [PPO](examples/ppo/ppo.py)
- [SAC](examples/sac/sac.py)
- [REDQ](examples/redq/redq.py)

and many more to come!

We also provide [tutorials and demos](tutorials) that give a sense of what the
library can do.

## Installation
Create a conda environment where the packages will be installed.

```
conda create --name torch_rl python=3.9
conda activate torch_rl
```

Depending on the use of functorch that you want to make, you may want to install the latest (nightly) pytorch release or the latest stable version of pytorch.
See [here](https://pytorch.org/get-started/locally/) for a more detailed list of commands, including `pip3` or windows/OSX compatible installation commands:

**Stable**

```
# For CUDA 11.3
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
# For CUDA 11.6
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
# For CPU-only build
conda install pytorch torchvision cpuonly -c pytorch

# For torch 1.12 (and not above), one should install functorch separately:
pip3 install functorch
```

**Nightly**
```
# For CUDA 11.6
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch-nightly -c nvidia
# For CUDA 11.7
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch-nightly -c nvidia
# For CPU-only build
conda install pytorch torchvision torchaudio cpuonly -c pytorch-nightly
```

`functorch` is included in the nightly PyTorch package, so no need to install it separately.

For M1 Mac users, if the above commands do not work, you can build torch from source by following [this guide](https://github.com/pytorch/pytorch#from-source).

**Torchrl**

You can install the **latest stable release** by using
```
pip3 install torchrl
```
This should work on linux and MacOs (not M1). For Windows and M1/M2 machines, one
should install the library locally (see below).

The **nightly build** can be installed via 
```
pip install torchrl-nightly
```

To install extra dependencies, call
```
pip3 install "torchrl[atari,dm_control,gym_continuous,rendering,tests,utils]"
```
or a subset of these.

Alternatively, as the library is at an early stage, it may be wise to install
it in develop mode as this will make it possible to pull the latest changes and
benefit from them immediately.
Start by cloning the repo:
```
git clone https://github.com/pytorch/rl
```

Go to the directory where you have cloned the torchrl repo and install it
```
cd /path/to/torchrl/
pip install -e .
```

On M1 machines, this should work out-of-the-box with the nightly build of PyTorch.
If the generation of this artifact in MacOs M1 doesn't work correctly or in the execution the message
`(mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64e'))` appears, then try

```
ARCHFLAGS="-arch arm64" python setup.py develop
```

To run a quick sanity check, leave that directory (e.g. by executing `cd ~/`)
and try to import the library.
```
python -c "import torchrl"
```
This should not return any warning or error.

**Optional dependencies**

The following libraries can be installed depending on the usage one wants to
make of torchrl:
```
# diverse
pip3 install tqdm tensorboard "hydra-core>=1.1" hydra-submitit-launcher

# rendering
pip3 install moviepy

# deepmind control suite
pip3 install dm_control

# gym, atari games
pip3 install "gym[atari]" "gym[accept-rom-license]" pygame

# tests
pip3 install pytest pyyaml pytest-instafail

# tensorboard
pip3 install tensorboard

# wandb
pip3 install wandb
```

**Troubleshooting**

If a `ModuleNotFoundError: No module named ‘torchrl._torchrl` errors occurs,
it means that the C++ extensions were not installed or not found.
One common reason might be that you are trying to import torchrl from within the
git repo location. Indeed the following code snippet should return an error if
torchrl has not been installed in `develop` mode:
```
cd ~/path/to/rl/repo
python -c 'from torchrl.envs.libs.gym import GymEnv'
```
If this is the case, consider executing torchrl from another location.

On **MacOs**, we recommend installing XCode first.
With Apple Silicon M1 chips, make sure you are using the arm64-built python
(e.g. [here](https://betterprogramming.pub/how-to-install-pytorch-on-apple-m1-series-512b3ad9bc6)). Running the following lines of code

```
wget https://raw.githubusercontent.com/pytorch/pytorch/master/torch/utils/collect_env.py
python collect_env.py
```
should display
```
OS: macOS *** (arm64)
```
and not
```
OS: macOS **** (x86_64)
```

## Running examples
Examples are coded in a very similar way but the configuration may change from one algorithm to another (e.g. async/sync data collection, hyperparameters, ratio of model updates / frame etc.)

Check the [examples markdown](examples/EXAMPLES.md) directory for more details about handling the various configuration settings.

## Contributing

Internal collaborations to torchrl are welcome! Feel free to fork, submit issues and PRs.
You can checkout the detailed contribution guide [here](CONTRIBUTING.md).
As mentioned above, a list of open contributions can be found in [here](https://github.com/pytorch/rl/issues/509).

Contributors are recommended to install [pre-commit hooks](https://pre-commit.com/) (using `pre-commit install`). pre-commit will check for linting related issues when the code is commited locally. You can disable th check by appending `-n` to your commit command: `git commit -m <commit message> -n`


## Upcoming features

In the near future, we plan to:
- provide tutorials on how to design new actors or environment wrappers;
- implement IMPALA (as a distributed RL example) and Meta-RL algorithms;
- improve the tests, documentation and nomenclature.

We welcome any contribution, should you want to contribute to these new features
or any other, lister or not, in the issues section of this repository.

# License
TorchRL is licensed under the MIT License. See [LICENSE](LICENSE) for details.
