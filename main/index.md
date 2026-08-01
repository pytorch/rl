# TorchRL

[![_images/logo.png](_images/logo.png)](_images/logo.png)

TorchRL is an open-source Reinforcement Learning (RL) library for PyTorch.

You can install TorchRL directly from PyPI (see more about installation
instructions in the dedicated section below):

```
$ pip install torchrl
```

TorchRL provides pytorch and python-first, low and high level abstractions for RL that are intended to be efficient, modular, documented and properly tested.
The code is aimed at supporting research in RL. Most of it is written in python in a highly modular way, such that researchers can easily swap components, transform them or write new ones with little effort.

This repo attempts to align with the existing pytorch ecosystem libraries in that it has a "dataset pillar"
[(environments)](reference/envs_api.html#environment-api),
[transforms](reference/envs_transforms.html#transforms),
[models](reference/modules.html#ref-modules),
data utilities (e.g. collectors and containers), etc.
TorchRL aims at having as few dependencies as possible (python standard library, numpy and pytorch).
Common environment libraries (e.g. OpenAI gym) are only optional.

On the low-level end, torchrl comes with a set of highly reusable functionals
for [cost functions](reference/objectives.html#ref-objectives), [returns](reference/objectives_common.html#ref-returns) and data processing.

TorchRL aims at a high modularity and good runtime performance.

To read more about TorchRL philosophy and capabilities beyond this API reference,
check the [TorchRL paper](https://arxiv.org/abs/2306.00577).

# Installation

TorchRL releases are synced with PyTorch, so make sure you always enjoy the latest
features of the library with the [most recent version of PyTorch](https://pytorch.org/get-started/locally/) (although core features
are guaranteed to be backward compatible with pytorch>=2.0).
Nightly releases can be installed via

```
$ pip install tensordict-nightly
$ pip install torchrl-nightly
```

or via a `git clone` if you're willing to contribute to the library:

```
$ cd path/to/root
$ git clone https://github.com/pytorch/tensordict
$ git clone https://github.com/pytorch/rl
$ cd tensordict
$ python setup.py develop
$ cd ../rl
$ python setup.py develop
```

If you use `uv` and you installed a specific PyTorch build beforehand (e.g. a nightly wheel),
use `--no-deps` for the editable installs to prevent dependency re-resolution (and potential PyTorch downgrades):

```
$ cd path/to/root
$ git clone https://github.com/pytorch/tensordict
$ git clone https://github.com/pytorch/rl
$ cd tensordict
$ uv pip install --no-deps -e .
$ cd ../rl
$ uv pip install --no-deps -e .
```

# Getting started

A series of quick tutorials to get ramped up with the basic features of the
library. If you're in a hurry, you can start by
[the last item of the series](tutorials/getting-started-5.html#gs-first-training)
and navigate to the previous ones whenever you want to learn more!

- [Get started with Environments, TED and transforms](tutorials/getting-started-0.html)
- [Get started with TorchRL's modules](tutorials/getting-started-1.html)
- [Getting started with model optimization](tutorials/getting-started-2.html)
- [Get started with data collection and storage](tutorials/getting-started-3.html)
- [Get started with logging](tutorials/getting-started-4.html)
- [Get started with your own first training loop](tutorials/getting-started-5.html)

# Tutorials

## Basics

- [Reinforcement Learning (PPO) with TorchRL Tutorial](tutorials/coding_ppo.html)
- [Pendulum: Writing your environment and transforms with TorchRL](tutorials/pendulum.html)
- [Introduction to TorchRL](tutorials/torchrl_demo.html)

## Intermediate

- [Multi-Agent Reinforcement Learning (PPO) with TorchRL Tutorial](tutorials/multiagent_ppo.html)
- [TorchRL envs](tutorials/torchrl_envs.html)
- [Using pretrained models](tutorials/pretrained_models.html)
- [Recurrent DQN: Training recurrent policies](tutorials/dqn_with_rnn.html)
- [Recurrent training on sequence batches](tutorials/recurrent_sequence_training.html)
- [MuJoCo scripted manipulation with human-readable robot actions](tutorials/mujoco_cube_bowl_macros.html)
- [Collectors Deep Dive: Trajectory Assembly](tutorials/collector_trajectory_assembly.html)
- [Using the Evaluator](tutorials/evaluator.html)
- [Using Replay Buffers](tutorials/rb_tutorial.html)
- [Memory-Efficient RL Training](tutorials/memory_efficient_rl.html)
- [Exporting TorchRL modules](tutorials/export.html)
- [Vision-Language-Action (VLA) policies with TorchRL](tutorials/vla.html)

## Advanced

- [Competitive Multi-Agent Reinforcement Learning (DDPG) with TorchRL Tutorial](tutorials/multiagent_competitive_ddpg.html)
- [Task-specific policy in multi-task environments](tutorials/multi_task.html)
- [TorchRL objectives: Coding a DDPG loss](tutorials/coding_ddpg.html)
- [TorchRL trainer: A DQN example](tutorials/coding_dqn.html)

# References

- [API Reference](reference/index.html)

- [torchrl.collectors package](reference/collectors.html)

- [Backend selection](reference/collectors.html#backend-selection)
- [Process collection](reference/collectors.html#process-collection)
- [Key Features](reference/collectors.html#key-features)
- [Collection hooks](reference/collectors.html#collection-hooks)
- [Quick Example](reference/collectors.html#quick-example)
- [Removed legacy names](reference/collectors.html#removed-legacy-names)
- [Documentation Sections](reference/collectors.html#documentation-sections)
- [Checkpointing](reference/checkpoint.html)

- [Basic usage](reference/checkpoint.html#basic-usage)
- [State-dict payload formats](reference/checkpoint.html#state-dict-payload-formats)
- [Custom components](reference/checkpoint.html#custom-components)
- [Compatibility](reference/checkpoint.html#compatibility)
- [API](reference/checkpoint.html#api)
- [torchrl.data package](reference/data.html)

- [Key Features](reference/data.html#key-features)
- [Quick Example](reference/data.html#quick-example)
- [CUDA prioritized replay buffers](reference/data.html#cuda-prioritized-replay-buffers)
- [Documentation Sections](reference/data.html#documentation-sections)
- [Data layout: contiguous trajectories](reference/data_layout.html)

- [Trajectory boundary keys](reference/data_layout.html#trajectory-boundary-keys)
- [Trajectory boundaries: recovering episodes from storage](reference/data_layout.html#trajectory-boundaries-recovering-episodes-from-storage)
- [The replay buffer `ndim` arg and why it doesn't multi-process well](reference/data_layout.html#the-replay-buffer-ndim-arg-and-why-it-doesn-t-multi-process-well)
- [The buffer-to-collector handoff: complete-trajectory writes](reference/data_layout.html#the-buffer-to-collector-handoff-complete-trajectory-writes)
- [SliceSampler: variable-length contiguous slices](reference/data_layout.html#slicesampler-variable-length-contiguous-slices)
- [Auto-discoverability for recurrent policies](reference/data_layout.html#auto-discoverability-for-recurrent-policies)
- [Legacy: `split_trajectories`](reference/data_layout.html#legacy-split-trajectories)
- [Narrow canonicalization for recurrent inputs](reference/data_layout.html#narrow-canonicalization-for-recurrent-inputs)
- [See also](reference/data_layout.html#see-also)
- [torchrl.envs package](reference/envs.html)

- [Key Features](reference/envs.html#key-features)
- [Quick Example](reference/envs.html#quick-example)
- [Documentation Sections](reference/envs.html#documentation-sections)
- [LLM Interface](reference/llms.html)

- [Key Components](reference/llms.html#key-components)
- [Quick Example](reference/llms.html#quick-example)
- [Documentation Sections](reference/llms.html#documentation-sections)
- [Environments](reference/llms.html#environments)
- [Objectives](reference/llms.html#objectives)
- [torchrl.modules package](reference/modules.html)

- [Key Features](reference/modules.html#key-features)
- [Quick Example](reference/modules.html#quick-example)
- [Documentation Sections](reference/modules.html#documentation-sections)
- [torchrl.objectives package](reference/objectives.html)

- [Key Features](reference/objectives.html#key-features)
- [Quick Example](reference/objectives.html#quick-example)
- [Documentation Sections](reference/objectives.html#documentation-sections)
- [Rendering applications](reference/render.html)

- [Core API](reference/render.html#core-api)
- [Configuration and results](reference/render.html#configuration-and-results)
- [Backends](reference/render.html#backends)
- [Lower-level helpers](reference/render.html#lower-level-helpers)
- [Service Registry](reference/services.html)

- [Service owners and clients](reference/services.html#service-owners-and-clients)
- [Overview](reference/services.html#overview)
- [Basic Usage](reference/services.html#basic-usage)
- [Python Executor Service](reference/services.html#python-executor-service)
- [Advanced Usage](reference/services.html#advanced-usage)
- [API Reference](reference/services.html#api-reference)
- [Best Practices](reference/services.html#best-practices)
- [Examples](reference/services.html#examples)
- [See Also](reference/services.html#see-also)
- [Designing Training Applications with Services](reference/services_workflow.html)

- [Scoped backend defaults](reference/services_workflow.html#scoped-backend-defaults)
- [Owners and clients](reference/services_workflow.html#owners-and-clients)
- [Placement does not define communication](reference/services_workflow.html#placement-does-not-define-communication)
- [Choosing a payload transport](reference/services_workflow.html#choosing-a-payload-transport)
- [Preserving domain APIs](reference/services_workflow.html#preserving-domain-apis)
- [Completion and failure semantics](reference/services_workflow.html#completion-and-failure-semantics)
- [Lifecycle belongs to the owner](reference/services_workflow.html#lifecycle-belongs-to-the-owner)
- [Integrations accept owners when they can](reference/services_workflow.html#integrations-accept-owners-when-they-can)
- [Environments are execution resources, not shared services](reference/services_workflow.html#environments-are-execution-resources-not-shared-services)
- [Discovery is optional](reference/services_workflow.html#discovery-is-optional)
- [Design compromises](reference/services_workflow.html#design-compromises)
- [Runnable examples](reference/services_workflow.html#runnable-examples)
- [Distributed transport implementation notes](reference/services_workflow.html#distributed-transport-implementation-notes)
- [torchrl.trainers package](reference/trainers.html)

- [Key Features](reference/trainers.html#key-features)
- [Quick Example](reference/trainers.html#quick-example)
- [Documentation Sections](reference/trainers.html#documentation-sections)
- [torchrl._utils package](reference/utils.html)

- [implement_for](reference/generated/torchrl.implement_for.html)
- [set_auto_unwrap_transformed_env](reference/generated/torchrl.set_auto_unwrap_transformed_env.html)
- [auto_unwrap_transformed_env](reference/generated/torchrl.auto_unwrap_transformed_env.html)
- [Memory profiling](reference/utils.html#memory-profiling)
- [Vision-Language-Action (VLA)](reference/vla.html)

- [Canonical TensorDict schema](reference/vla.html#canonical-tensordict-schema)
- [Data and metadata](reference/vla.html#data-and-metadata)
- [Transforms](reference/vla.html#transforms)
- [Action representations](reference/vla.html#action-representations)
- [Image preprocessing](reference/vla.html#image-preprocessing)
- [Policy and environment contract](reference/vla.html#policy-and-environment-contract)
- [Inference loop sketch](reference/vla.html#inference-loop-sketch)
- [Training loop sketch](reference/vla.html#training-loop-sketch)
- [Policies](reference/vla.html#policies)
- [Objectives](reference/vla.html#objectives)
- [TorchRL Configuration System](reference/config.html)

- [Quick Start with a Simple Example](reference/config.html#quick-start-with-a-simple-example)
- [Configuration Categories and Groups](reference/config.html#configuration-categories-and-groups)
- [More Complex Example: Parallel Environment with Transforms](reference/config.html#more-complex-example-parallel-environment-with-transforms)
- [Getting Available Options](reference/config.html#getting-available-options)
- [Complete Training Example](reference/config.html#complete-training-example)
- [Running Experiments](reference/config.html#running-experiments)
- [Configuration Store Implementation Details](reference/config.html#configuration-store-implementation-details)
- [Available Configuration Classes](reference/config.html#available-configuration-classes)
- [Creating Custom Configurations](reference/config.html#creating-custom-configurations)
- [Best Practices](reference/config.html#best-practices)
- [Supported Algorithms](reference/config.html#supported-algorithms)
- [Profiling collectors and envs](reference/profiling.html)

- [Enabling profiling](reference/profiling.html#enabling-profiling)
- [Driving a torch.profiler trace from the driver](reference/profiling.html#driving-a-torch-profiler-trace-from-the-driver)
- [What gets instrumented](reference/profiling.html#what-gets-instrumented)
- [Capturing a trace](reference/profiling.html#capturing-a-trace)
- [Multi-process and Ray](reference/profiling.html#multi-process-and-ray)
- [Performance impact](reference/profiling.html#performance-impact)
- [Glossary](reference/glossary.html)

- [See also](reference/glossary.html#see-also)

# Knowledge Base

- [Knowledge Base](reference/knowledge_base.html)

- [Contributing](reference/knowledge_base.html#contributing)
- [Things to consider when debugging RL](reference/generated/knowledge_base/DEBUGGING_RL.html)
- [Installing dm_control](reference/generated/knowledge_base/DM_CONTROL_INSTALLATION.html)
- [Flaky Test Resolution Guide](reference/generated/knowledge_base/FLAKY_TESTS.html)
- [Working with gym](reference/generated/knowledge_base/GYM.html)
- [Working with `habitat-lab`](reference/generated/knowledge_base/HABITAT.html)
- [IsaacLab Guide](reference/generated/knowledge_base/ISAACLAB.html)
- [Working with MuJoCo-based environments](reference/generated/knowledge_base/MUJOCO_INSTALLATION.html)
- [Common PyTorch errors and solutions](reference/generated/knowledge_base/PRO-TIPS.html)
- [Useful resources](reference/generated/knowledge_base/RESOURCES.html)
- [Versioning Issues](reference/generated/knowledge_base/VERSIONING_ISSUES.html)
- [Customising Video Renders](reference/generated/knowledge_base/VIDEO_CUSTOMISATION.html)

# Indices and tables

- [Index](genindex.html)
- [Module Index](py-modindex.html)
- [Search Page](search.html)