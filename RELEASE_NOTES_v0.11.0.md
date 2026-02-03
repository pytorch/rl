# TorchRL v0.11.0 Release Notes

## Highlights

- **Dreamer overhaul** - Comprehensive improvements to Dreamer world model training: async collectors with profiling, RSSM fixes (scan mode, noise injection, explicit dimensions), torch.compile compatibility for value functions and TDLambda estimator, optimized DreamerEnv to avoid CUDA syncs, and updated sota-implementation with better configs. @vmoens
- **Weight synchronization schemes** - New modular weight sync infrastructure (`torchrl.weight_update`) with SharedMem, MultiProcess, and vLLM-specific (NCCL, double-buffer) schemes. Collectors now integrate seamlessly with weight sync schemes for distributed training. @vmoens
- **Major collector refactor** - The collector codebase has been completely restructured. The monolithic `collectors.py` is now split into focused modules (`_single.py`, `_multi_base.py`, `_multi_sync.py`, `_multi_async.py`, `_runner.py`, `base.py`), with cleaner separation of concerns. ([#3233](https://github.com/pytorch/rl/pull/3233)) @vmoens
- **LLM objectives: DAPO & CISPO** - New DAPO (Direct Advantage Policy Optimization) and CISPO (Clipped Importance Sampling Policy Optimization) algorithms for LLM training. @vmoens
- **Trainer infrastructure** - New SAC Trainer, configuration system for algorithms, timing utilities, and async collection support within trainers. @vmoens
- **Tool services** - New tool service infrastructure for LLM agents with Python executor, MCP tools, and web search capabilities. @vmoens
- **Deprecated APIs removed** - All deprecation warnings from v0.10 have been promoted to hard errors for v0.11. ([#3369](https://github.com/pytorch/rl/pull/3369)) @vmoens
- **New environment backends** - Added Procgen environments support with a new `ProcgenEnv` wrapper. ([#3331](https://github.com/pytorch/rl/pull/3331)) @ParamThakkar123
- **Multi-env execution** - GymEnv, BraxEnv, and DMControlEnv now support a `num_envs`/`num_workers` parameter to run multiple environments in a single call via `ParallelEnv`. ([#3343](https://github.com/pytorch/rl/pull/3343), [#3370](https://github.com/pytorch/rl/pull/3370), [#3337](https://github.com/pytorch/rl/pull/3337)) @ParamThakkar123

## Installation

```bash
pip install torchrl==0.11.0
```

---

## Breaking Changes

- **[v0.11] Remove deprecated features and replace warnings with errors** ([#3369](https://github.com/pytorch/rl/pull/3369)) @vmoens
  - Removes deprecated `KLRewardTransform` from `transforms/llm.py` (use `torchrl.envs.llm.KLRewardTransform`)
  - Removes `LogReward` and `Recorder` classes from trainers (use `LogScalar` and `LogValidationReward`)
  - Removes `unbatched_*_spec` properties from VmasWrapper/VmasEnv (use `full_*_spec_unbatched`)
  - Deletes deprecated `rlhf.py` modules (`data/rlhf.py`, `envs/transforms/rlhf.py`, `modules/models/rlhf.py`)
  - Removes `replay_buffer_chunk` parameter from MultiCollector
  - Replaces `minimum`/`maximum` deprecation warnings with `TypeError` in Bounded spec
  - Replaces `critic_coef`/`entropy_coef` deprecation warnings with `TypeError` in PPO and A2C losses

- **[Major] Major refactoring of collectors** ([#3233](https://github.com/pytorch/rl/pull/3233)) @vmoens
  - Splits the 5000+ line `collectors.py` into focused modules for single/multi sync/async collectors
  - Creates new `_constants.py`, `_runner.py`, `base.py` modules
  - Introduces cleaner weight synchronization scheme integration
  - Improves test coverage for multi-device and shared-device weight updates
  - Some internal APIs have changed; external API remains compatible

---

## Dreamer World Model Improvements

These changes significantly improve Dreamer training performance, torch.compile compatibility, and usability. @vmoens

- **[Feature] Refactor Dreamer training with async collectors, profiling, and improved config** (cc917bae)
  - Major overhaul of the Dreamer sota-implementation with async data collection
  - Adds profiling support for performance analysis
  - Improved configuration with better defaults and documentation
  - Updated README with detailed usage instructions

- **[Refactor] Dreamer implementation updates** (3ab4b306)
  - Refactors Dreamer training script for better maintainability
  - Updates config.yaml with improved hyperparameters
  - Enhances dreamer_utils.py with additional helper functions

- **[Feature] Add noise argument and scan mode to RSSMRollout** (8653b6e6)
  - Adds `noise` argument to control stochastic sampling during rollout
  - Implements scan mode for efficient sequential processing
  - 109 lines added to model_based.py for improved RSSM flexibility

- **[Feature] Add explicit dimensions and device support to RSSM modules** (f350fe06)
  - Adds explicit dimension handling for batch, time, and feature dims
  - Improves device placement for RSSM components

- **[Feature] Logging & RSSM fixes** (d0829794)
  - Fixes RSSM module behavior and adds logging improvements
  - Updates batched_envs.py and wandb logger

- **[Refactor] Use compile-aware helpers in Dreamer objectives** (d8c3887d)
  - Updates Dreamer objectives to use torch.compile-compatible helpers
  - Improves performance when using torch.compile

- **[BugFix] Optimize DreamerEnv to avoid CUDA sync in done checks** (d57fdecf)
  - Eliminates unnecessary CUDA synchronizations in done flag checking
  - Significant performance improvement for GPU-based Dreamer training

- **[BugFix] Fix ModelBasedEnvBase for torch.compile compatibility** (7ff663d6)
  - Makes ModelBasedEnvBase compatible with torch.compile

- **[Feature] Add allow_done_after_reset parameter to ModelBasedEnvBase** (24ae0424)
  - Adds flexibility for environments that may signal done immediately after reset

- **[BugFix] Final polish for Dreamer utils and Collector tests** (5aefdd5c)
  - Final cleanup and polish for Dreamer implementation

---

## Weight Synchronization Schemes

New modular infrastructure for weight synchronization between training and inference workers. @vmoens

- **[Feature] Weight Synchronization Schemes - Core Infrastructure** (e8f6fa5a)
  - New `torchrl.weight_update` module with 2400+ lines of weight sync infrastructure
  - `SharedMemWeightSyncScheme`: Uses shared memory for fast intra-node sync
  - `MultiProcessWeightSyncScheme`: Uses multiprocessing queues for cross-process sync
  - Comprehensive documentation and examples in `docs/source/reference/collectors.rst`

- **[Feature] vLLM Weight Synchronization Schemes** (d0c8b7e9)
  - `VllmNCCLWeightSyncScheme`: NCCL-based weight sync for vLLM distributed inference
  - `VllmDoubleBufferWeightSyncScheme`: Double-buffered async weight updates
  - 1876 lines of vLLM-specific weight sync code

- **[Feature] Collectors - Weight Sync Scheme Integration** (a7707ca0)
  - Integrates weight sync schemes into collector infrastructure
  - Updates GRPO and expert-iteration implementations to use new schemes
  - Adds examples for multi-weight update patterns

- **[Refactor] Weight sync schemes refactor** (ae0ae064)
  - Refines weight sync API and adds additional schemes
  - Improves test coverage with 342+ new test lines

---

## LLM Training: DAPO & CISPO

New policy optimization algorithms for LLM fine-tuning. @vmoens

- **[Feature] DAPO** (9d5c2767)
  - Implements Direct Advantage Policy Optimization for LLM training
  - Adds DAPO-specific loss computation to `torchrl/objectives/llm/grpo.py`

- **[Feature] CISPO** (ed0d8dc6)
  - Implements Clipped Importance Sampling Policy Optimization
  - Alternative to PPO/GRPO with different clipping strategy

- **[Refactor] Refactor GRPO as a separate class** (2bc3cb70)
  - Separates GRPO implementation for better modularity

---

## Trainer Infrastructure

New trainer algorithms, configuration system, and utilities. @vmoens

- **[Trainers] SAC Trainer and algorithms** (02d4bfd1)
  - New SAC Trainer implementation with 786 lines of code
  - Complete sota-implementation in `sota-implementations/sac_trainer/`
  - Trainer configuration via YAML files

- **[Feature] Trainer Algorithms - Configuration System** (6bc201a5)
  - New configuration system in `torchrl/trainers/algorithms/configs/`
  - Configs for collectors, data, modules, objectives, transforms, weight sync schemes
  - Enables hydra-style configuration composition

- **[Feature] Trainer Infrastructure - Timing and Utilities** (dc215236)
  - Adds timing utilities to trainer infrastructure
  - 263 lines of enhanced trainer functionality

- **[Feature] Async collection within trainers** (5f1eb2c5)
  - Enables asynchronous data collection during training
  - Improves training throughput

- **[Feature] PPO Trainer Updates** (129f3d50)
  - Updates to PPO trainer with new features

---

## Tool Services for LLM Agents

New infrastructure for tool-augmented LLM agents. @vmoens

- **[Feature] Tool services** (9ca0e407)
  - New `torchrl/services/` module for tool execution
  - Python executor service for safe code execution
  - MCP (Model Context Protocol) tool integration
  - Web search tool example
  - 609 lines of documentation in `docs/source/reference/services.rst`
  - Comprehensive test coverage in `test/test_services.py`

- **[Feature] Transform Module - ModuleTransform and Ray Service Refactor** (7b85c716)
  - New `ModuleTransform` for applying nn.Modules as transforms
  - Refactored Ray service integration
  - Moves `ray_service.py` to `torchrl/envs/transforms/`

---

## torch.compile Compatibility

Fixes to enable torch.compile with various TorchRL components. @vmoens

- **[BugFix] Fix value functions for torch.compile compatibility** (3bdc7b18)
  - 295 lines of new tests in `test/compile/test_value.py`
  - Fixes value function implementations for compile compatibility

- **[BugFix] Fix TDLambdaEstimator for torch.compile compatibility** (11e22ee9)
  - Updates TDLambdaEstimator to work with torch.compile

- **[Feature] Add compile-aware timing and profiling helpers** (764d9f71)
  - New utilities for profiling compiled code

- **[Refactor] Add compile-aware timing and profiling helpers** (c82447e6)
  - Performance utilities that work correctly under torch.compile

---

## Features

- **[Feature] Added EXP3 Scoring function** ([#3013](https://github.com/pytorch/rl/pull/3013)) @ParamThakkar123
  - Implements the EXP3 (Exponential-weight algorithm for Exploration and Exploitation) scoring function for MCTS
  - Adds `torchrl/data/map/` module with hash-based storage, query utilities, and tree structures for efficient state lookups

- **[Feature] Add num_workers parameter to BraxEnv** ([#3370](https://github.com/pytorch/rl/pull/3370)) @ParamThakkar123
  - Allows running multiple Brax environments in parallel via `ParallelEnv` by specifying `num_workers > 1`
  - Consistent API with GymEnv and DMControlEnv multi-env support

- **[Feature] Added num_envs parameter in GymEnv** ([#3343](https://github.com/pytorch/rl/pull/3343)) @ParamThakkar123
  - When `num_envs > 1`, GymEnv automatically returns a lazy `ParallelEnv` wrapping multiple environment instances
  - Simplifies multi-env setup without manually constructing ParallelEnv

- **[Environments] Added Procgen environments** ([#3331](https://github.com/pytorch/rl/pull/3331)) @ParamThakkar123
  - New `ProcgenWrapper` and `ProcgenEnv` classes to wrap OpenAI Procgen environments
  - Converts Procgen observations to TorchRL-compliant TensorDict outputs
  - Supports all 16 Procgen game environments (coinrun, starpilot, etc.)

- **[Feature] Loss make_value_estimator takes a ValueEstimatorBase class** ([#3336](https://github.com/pytorch/rl/pull/3336)) @ParamThakkar123
  - Allows passing a `ValueEstimatorBase` class directly to `make_value_estimator()` for more flexible value estimation configuration
  - Adds tests and documentation for the new API

- **[Feature] Make custom_range public in ActionDiscretizer** ([#3333](https://github.com/pytorch/rl/pull/3333)) @vmoens
  - Exposes `custom_range` parameter in ActionDiscretizer for user-defined discretization ranges

- **[Feature] Added num_envs parameter in DMControlEnv** ([#3337](https://github.com/pytorch/rl/pull/3337)) @ParamThakkar123
  - Similar to GymEnv, allows running multiple DMControl environments via `num_workers` parameter
  - Returns a lazy `ParallelEnv` when `num_workers > 1`

- **[Feature] Auto-configure exploration module specs from environment** ([#3317](https://github.com/pytorch/rl/pull/3317)) @bsprenger
  - Collectors now automatically configure exploration module specs (like `AdditiveGaussianModule`) from the environment's action spec
  - Adds `set_exploration_modules_spec_from_env()` utility function
  - Reduces boilerplate when using exploration modules with delayed spec initialization

- **[Feature] Add NPU Support for Single Agent** ([#3229](https://github.com/pytorch/rl/pull/3229)) @lowdy1
  - Adds Huawei NPU (Ascend) device support for single-agent training
  - Updates device handling logic to recognize NPU devices

- **[Feature] Add support for `trackio`** ([#3196](https://github.com/pytorch/rl/pull/3196)) @Xmaster6y
  - Integrates with trackio for experiment tracking and logging
  - New `TrackioLogger` class for seamless integration

- **[Feature] Add a new Trainer hook point `process_loss`** ([#3259](https://github.com/pytorch/rl/pull/3259)) @Xmaster6y
  - Adds a new hook point in the Trainer that runs after loss computation but before backward pass
  - Useful for loss modification, logging, or gradient accumulation strategies

- **[Feature] Ensure `MultiSyncDataCollectors` returns data ordered by worker id** ([#3243](https://github.com/pytorch/rl/pull/3243)) @LCarmi
  - Data from MultiSyncDataCollectors is now consistently ordered by worker ID
  - Makes debugging and analysis easier when working with multi-worker collectors

- **[Feature] Add TanhModuleConfig** ([#3255](https://github.com/pytorch/rl/pull/3255)) @bsprenger
  - Adds configuration class for TanhModule to support serialization and hydra-style configs

- **[Feature] add TensorDictSequentialConfig** ([#3248](https://github.com/pytorch/rl/pull/3248)) @bsprenger
  - Adds configuration class for TensorDictSequential modules

- **[Feature] Weight loss outputs when using prioritized sampler** ([#3235](https://github.com/pytorch/rl/pull/3235)) @vmoens
  - Loss modules now properly weight outputs when using prioritized replay buffers
  - Adds `importance_key` parameter to loss functions (DQN, DDPG, SAC, TD3, TD3+BC)
  - When importance weights are present in the sampled data, losses are automatically weighted
  - Adds comprehensive tests for prioritized replay buffer integration with all loss functions

- **[Feature] Composite specs can create named tensors with 'zero' and 'rand'** ([#3214](https://github.com/pytorch/rl/pull/3214)) @louisfaury
  - Composite specs now propagate dimension names when creating tensors via `zero()` and `rand()`

- **[Feature] Enable storing rollouts on a different device** ([#3199](https://github.com/pytorch/rl/pull/3199)) @Xmaster6y
  - Collectors can now store rollouts on a different device than the execution device
  - Useful for CPU storage while running on GPU

- **[Feature] Named dims in Composite** ([#3174](https://github.com/pytorch/rl/pull/3174)) @vmoens
  - Adds named dimension support to Composite specs via `names` property and `refine_names()` method
  - Enables better integration with PyTorch's named tensors
  - Supports ellipsis notation for partial name specification

---

## Bug Fixes

- **[BugFix] Fix TransformersWrapper ChatHistory.full not being set** ([#3375](https://github.com/pytorch/rl/pull/3375)) @vmoens

- **[BugFix] Fix AsyncEnvPool + LLMCollector with yield_completed_trajectories** ([#3373](https://github.com/pytorch/rl/pull/3373)) @vmoens
  - Fixes interaction between AsyncEnvPool and LLMCollector when yielding completed trajectories

- **[BugFix] Fix LLM test failures** ([#3360](https://github.com/pytorch/rl/pull/3360)) @vmoens
  - Comprehensive fixes for LLM-related test failures across multiple modules

- **[BugFix] Fixed MultiSyncCollector set_seed and split_trajs issue** ([#3352](https://github.com/pytorch/rl/pull/3352)) @ParamThakkar123
  - Fixes seed propagation and trajectory splitting in MultiSyncDataCollector

- **[BugFix] Fix VecNormV2 device gathering** ([#3368](https://github.com/pytorch/rl/pull/3368)) @vmoens
  - Fixes device handling when gathering statistics in VecNormV2 transform

- **[BugFix] Fix AsyncEnv - LLMCollector integration** ([#3365](https://github.com/pytorch/rl/pull/3365)) @vmoens
  - Resolves integration issues between async environments and LLM collectors

- **[BugFix] Fix VecNormV2 GPU device handling for stateful mode** ([#3364](https://github.com/pytorch/rl/pull/3364)) @BY571
  - Fixes GPU device handling in VecNormV2 when using stateful normalization mode

- **[BugFix] Fix Ray collector iterator bug and process group cleanup** ([#3363](https://github.com/pytorch/rl/pull/3363)) @vmoens
  - Fixes iterator behavior in Ray-based collectors and ensures proper cleanup of process groups

- **[BugFix] Fix LLM CI by replacing uvx with uv package** ([#3356](https://github.com/pytorch/rl/pull/3356)) @vmoens

- **[Bugfix] Wrong minari download first element** ([#3106](https://github.com/pytorch/rl/pull/3106)) @marcosgalleterobbva
  - Fixes incorrect first element handling when downloading Minari datasets

- **[BugFix,Test] Fix envpool failing tests** ([#3345](https://github.com/pytorch/rl/pull/3345)) @vmoens

- **[BugFix] Fixed ParallelEnv + aSyncDataCollector / MultiSyncDataCollector not working if replay_buffer is given** ([#3341](https://github.com/pytorch/rl/pull/3341)) @ParamThakkar123
  - Fixes compatibility issue when using ParallelEnv with collectors that have a replay buffer attached

- **[BugFix] treat 1-D MultiDiscrete as MultiCategorical and accept flattened masks** ([#3342](https://github.com/pytorch/rl/pull/3342)) @ParamThakkar123
  - Improves MultiDiscrete action space handling to accept flattened action masks

- **[BugFix] Fixes register_save_hook bug** ([#3340](https://github.com/pytorch/rl/pull/3340)) @ParamThakkar123
  - Fixes save hook registration in replay buffers

- **[BugFix,Test] recompiles with string failure** ([#3338](https://github.com/pytorch/rl/pull/3338)) @vmoens

- **[BugFix] Fix Safe modules annotation and doc for losses** ([#3334](https://github.com/pytorch/rl/pull/3334)) @vmoens

- **[BugFix] Fix ray modules circular import** ([#3319](https://github.com/pytorch/rl/pull/3319)) @vmoens

- **[BugFix] Fix SACLoss target_entropy="auto" ignoring action space dimensionality** ([#3292](https://github.com/pytorch/rl/pull/3292)) @vmoens
  - Fixes automatic entropy target computation to properly account for action space dimensions

- **[BugFix] Fix agent_dim in multiagent nets & account for neg dims** ([#3290](https://github.com/pytorch/rl/pull/3290)) @vmoens
  - Fixes agent dimension handling in multi-agent networks, including support for negative dimension indices

- **[BugFix] Added a missing .to(device) call in _from_transformers_generate_history** ([#3289](https://github.com/pytorch/rl/pull/3289)) @michalgregor

- **[BugFix] Fix old pytorch dependencies** ([#3266](https://github.com/pytorch/rl/pull/3266)) @vmoens
  - Updates compatibility shims for older PyTorch versions

- **[BugFix] RSSMRollout not advancing state/belief across time steps in Dreamer** ([#3236](https://github.com/pytorch/rl/pull/3236)) @cmdout
  - Fixes critical bug where RSSM rollout was not properly advancing hidden states in Dreamer world model

- **[Doc,BugFix] Fix doc and collectors** ([#3250](https://github.com/pytorch/rl/pull/3250)) @vmoens

- **[BugFix] use correct field names in InitTrackerConfig** ([#3245](https://github.com/pytorch/rl/pull/3245)) @bsprenger

- **[BugFix] Use torch.zeros for argument in torch.where** ([#3239](https://github.com/pytorch/rl/pull/3239)) @sebimarkgraf

- **[BugFix] Fix wrong assertion about collector and buffer** ([#3176](https://github.com/pytorch/rl/pull/3176)) @vmoens

- **[BugFix] AttributeError in accept_remote_rref_udf_invocation** ([#3168](https://github.com/pytorch/rl/pull/3168)) @vmoens

### Bug Fixes (ghstack commits without PR numbers)

- **[BugFix] Final polish for Collector and Exploration** (558396d7) @vmoens
  - Final fixes for collector and exploration module interactions

- **[BugFix] Collector robustness & Async fixes** (cadf23bc) @vmoens
  - Improves collector robustness and fixes async collection issues

- **[BugFix] TensorStorage key filtering** (0667a580) @vmoens
  - Fixes key filtering in TensorStorage

- **[BugFix] Replay Buffer prefetch & SliceSampler** (9d34dbe7) @vmoens
  - Fixes prefetching behavior and SliceSampler issues

- **[BugFix] Fix target_entropy computation for composite action specs** (06960c64) @vmoens
  - Correctly computes target entropy for composite action spaces

- **[BugFix] Fix device initialization in CrossQLoss.maybe_init_target_entropy** (416e4545) @vmoens
  - Fixes device placement for CrossQ loss entropy initialization

- **[BugFix] Fix schemes and refactor collectors to make them readable** (364e0386) @vmoens
  - Fixes weight sync schemes and improves collector code readability

- **[BugFix] Fix collector devices** (888095fd) @vmoens
  - Fixes device handling in collectors

- **[BugFix] Fix tests** (963fdd43) @vmoens

- **[BugFix] Fix GRPO tests and runs** (47ad9d8f) @vmoens

- **[BugFix] Handle Lock/RLock types in EnvCreator** (6985ca25) @vmoens
  - Properly handles threading locks in EnvCreator serialization

- **[BugFix] Defer filter_warnings import to avoid module load issues** (8ea954cd) @vmoens

- **[BugFix] Fix CUDA sync in forked subprocess** (4a2a2749) @vmoens
  - Fixes CUDA synchronization issues in forked processes

- **[BugFix] Fix unique ref to lambda func** (b6fe45ee) @vmoens

- **[BugFix] Add pybind11 check and Windows extension pattern fix** (5e89e4e7) @vmoens

---

## Additional Features (ghstack commits without PR numbers)

- **[Feature] Storage Shared Initialization for Multiprocessing** (61c178e5) @vmoens
  - Enables shared storage initialization for multiprocessing collectors
  - Updates distributed collector examples

- **[Feature] Memmap storage cleanup** (7f9ea748) @vmoens
  - Improves memmap storage cleanup and resource management

- **[Feature] Collector Profiling** (02ed47ed) @vmoens
  - Adds profiling capabilities to collectors for performance analysis

- **[Feature] auto_wrap_envs in PEnv** (d781f9e9) @vmoens
  - Automatic environment wrapping in ParallelEnv

- **[Feature] Auto-wrap lambda functions with EnvCreator for spawn compatibility** (d250c182) @vmoens
  - Automatically wraps lambda functions with EnvCreator for spawn multiprocessing compatibility

- **[Feature] Collectors' getattr_policy and getattr_env** (fbdbb617) @vmoens
  - Adds attribute access methods for policy and env in collectors

- **[Feature] track_policy_version in collectors.py** (a089cc48) @vmoens
  - Adds policy version tracking in collectors for distributed training

- **[Feature] Aggregation strategies** (0e6a3565) @vmoens
  - New aggregation strategies for multi-worker data collection

- **[Feature] kl_mask_threshold** (7ab48a4c) @vmoens
  - Adds KL divergence masking threshold for LLM training

- **[Feature] Add timing options** (13434ebb) @vmoens
  - Additional timing options for performance monitoring

- **[Feature] float32 patch** (01d2801b) @vmoens
  - Float32 precision handling improvements

- **[Feature] Support callable scale in IndependentNormal and TanhNormal distributions** (d3aba7d1) @vmoens
  - Allows scale parameter to be a callable in distribution modules

---

## Documentation

- **[Doc] Huge doc refactoring** (3d5dd1ac) @vmoens
  - Major documentation restructuring with new sections:
    - `collectors_basics.rst`, `collectors_single.rst`, `collectors_distributed.rst`
    - `collectors_weightsync.rst`, `collectors_replay.rst`
    - `data_datasets.rst`, `data_replaybuffers.rst`, `data_samplers.rst`
  - Adds pre-commit hook for Sphinx section underline checking

- **[Docs] Update LLM_TEST_ISSUES.md with fix status** ([#3374](https://github.com/pytorch/rl/pull/3374)) @vmoens

- **[Docs] Enable doc builds and tutorial runs** ([#3335](https://github.com/pytorch/rl/pull/3335)) @vmoens
  - Re-enables Sphinx-gallery tutorial execution in documentation builds
  - Adds process cleanup between tutorials to prevent resource leaks

---

## CI / Infrastructure

- **[CI] Speed up slow tests in tests-gpu/tests-cpu** ([#3395](https://github.com/pytorch/rl/pull/3395)) @vmoens
  - Optimizes slow test execution with better parallelization and test isolation

- **[CI] Bump version to 0.11.0** ([#3392](https://github.com/pytorch/rl/pull/3392)) @vmoens

- **[CI] Fix auto-label workflow for fork PRs** ([#3388](https://github.com/pytorch/rl/pull/3388)) @vmoens

- **[CI] Better release workflow** ([#3386](https://github.com/pytorch/rl/pull/3386)) @vmoens
  - Comprehensive release workflow with sanity checks, wheel collection, docs updates, and PyPI publishing
  - Adds dry-run mode for testing releases
  - Automatically updates stable docs symlink and versions.html

- **[CI] Fix auto-labelling** ([#3387](https://github.com/pytorch/rl/pull/3387)) @vmoens

- **[CI] Auto-tag PRs** ([#3381](https://github.com/pytorch/rl/pull/3381)) @vmoens
  - Adds automatic labeling for PRs based on changed files

- **[CI] Add release agent prompt for LLM-assisted releases** ([#3380](https://github.com/pytorch/rl/pull/3380)) @vmoens
  - Adds comprehensive release instructions for LLM-assisted release automation

- **[CI] Add release workflow with PyPI trusted publishing** ([#3379](https://github.com/pytorch/rl/pull/3379)) @vmoens
  - Implements PyPI trusted publishing via OIDC authentication
  - Eliminates need for PyPI API tokens in secrets

- **[CI] Add granular label support for environment and data workflow jobs** ([#3371](https://github.com/pytorch/rl/pull/3371)) @vmoens

- **[CI] Fix PettingZoo CI by updating Python 3.9 to 3.10** ([#3362](https://github.com/pytorch/rl/pull/3362)) @vmoens

- **[CI] Fix GenDGRL CI by adding missing requests dependency** ([#3361](https://github.com/pytorch/rl/pull/3361)) @vmoens

- **[CI] Fix IsaacLab tests by using explicit conda Python path** ([#3358](https://github.com/pytorch/rl/pull/3358)) @vmoens

- **[CI] Fix M1 build pip command not found** ([#3359](https://github.com/pytorch/rl/pull/3359)) @vmoens

- **[CI] Fix Windows build for free-threaded Python (3.13t, 3.14t)** ([#3357](https://github.com/pytorch/rl/pull/3357)) @vmoens

- **[CI,BugFix] Fix Habitat CI by upgrading to Python 3.10** ([#3346](https://github.com/pytorch/rl/pull/3346)) @vmoens
  - Upgrades Habitat CI to Python 3.10 and builds habitat-sim from source

- **[CI] Fix Jumanji CI by adding missing requests dependency** ([#3349](https://github.com/pytorch/rl/pull/3349)) @vmoens

- **[CI] Fix Chess CI by correcting test file path typo** ([#3353](https://github.com/pytorch/rl/pull/3353)) @vmoens

- **[CI] Skip Windows-incompatible tests in optional deps CI** ([#3348](https://github.com/pytorch/rl/pull/3348)) @vmoens

- **[CI] Fix GPU benchmark failures** ([#3347](https://github.com/pytorch/rl/pull/3347)) @vmoens

- **[CI] Upgrade doc python version** ([#3222](https://github.com/pytorch/rl/pull/3222)) @vmoens

- **[CI] Use pip install** ([#3200](https://github.com/pytorch/rl/pull/3200)) @vmoens
  - Migrates CI workflows from setup.py install to pip install

### CI / Infrastructure (ghstack commits without PR numbers)

- **[Setup] Python 3.14 in, python 3.9 out** (ff86ab78) @vmoens
  - Adds Python 3.14 support and removes Python 3.9

- **[CI] LLM tests integration** (cccfaa69) @vmoens
  - Integrates LLM tests into CI pipeline

- **[CI] Fix benchmarks for LLMs** (e7ec9c31) @vmoens

- **[CI] Test setup** (2644b899) @vmoens

- **[CI] Isolate distributed tests** (5c5992d9) @vmoens

- **[CI] Fix uv install with --no-deps** (546a1b73) @vmoens

- **[CI] Fix SOTA runs** (57000fcc) @vmoens

- **[CI] Fix libs** (071d079f) @vmoens

- **[CI] Fix missing librhash0 in doc CI** (8d0f0613) @vmoens

- **[CI] Add --upgrade flag for torch installs and install ffmpeg/xvfb** (ce9931943) @vmoens

- **[CI] Quiet coverage combine** (1ac0fd0a) @vmoens

- **[CI] Fix tensordict install in lib tests** (8f487d40) @vmoens

---

## Tests

- **[Tests] Check traj_ids shape with unbatched envs** ([#3393](https://github.com/pytorch/rl/pull/3393)) @vmoens

- **[Tests] Fix test isolation in test_set_gym_environments and related tests** ([#3382](https://github.com/pytorch/rl/pull/3382)) @vmoens
  - Improves test isolation to prevent cross-test contamination

- **[Test] Use mock Llama tokenizer instead of skipping gated model test** ([#3376](https://github.com/pytorch/rl/pull/3376)) @vmoens

- **[Test] Skip Llama test when tokenizer unavailable instead of xfail** ([#3372](https://github.com/pytorch/rl/pull/3372)) @vmoens

- **[Tests] Remove _utils_internal.py in tests** ([#3281](https://github.com/pytorch/rl/pull/3281)) @vmoens
  - Removes deprecated internal test utilities and updates tests to use public APIs

- **[Test] Check LinearizeReward obs transform** ([#3241](https://github.com/pytorch/rl/pull/3241)) @vmoens

- **[Test,Benchmark] Move mocking classes file and bench for non-tensor env** ([#3257](https://github.com/pytorch/rl/pull/3257)) @vmoens

- **[Tests] Fix vmas seeding test** ([#3210](https://github.com/pytorch/rl/pull/3210)) @matteobettini

- **[Tests] Reintroduce VMAS football** ([#3178](https://github.com/pytorch/rl/pull/3178)) @matteobettini

### Tests (ghstack commits without PR numbers)

- **[Test] Use spawn in older pytorch** (9b0492906) @vmoens

- **[Test] Fix test_num_threads that instantiate the env in the main process** (852dd61b) @vmoens

- **[Test] Use class references instead of lambdas in transform tests** (7dd1e61d) @vmoens

- **[Test] Add test for lambda wrapping in ParallelEnv** (4b2b2279) @vmoens

- **[Test] Ensure collector shutdown with try/finally** (aa2b0311) @vmoens

- **[Test] Add filterwarnings for unclosed resources** (b493ab34) @vmoens

- **[Test] Add retry decorator to flaky vecnorm test** (6d908b69) @vmoens

- **[Test] Simplify num_threads test to check threads in env factory** (295cc20e) @vmoens

- **[Test] Ignore script_method deprecation warning** (eaaf97a4) @vmoens

- **[Test] Fix failing 3.14 tests** (66a48f3c) @vmoens

- **[Test] Remove the forked decorator of the distributed checks** (4db59660) @vmoens

- **[Test] Test RB+Isaac+Ray** (a0d650f9) @vmoens

- **[Test] Fix flaky parallel test** (4f013a81) @vmoens

- **[Test] Fix vc1** (bd93f136) @vmoens

- **[Quality] Fix flaky test** (03ffc820) @vmoens

---

## Refactors / Maintenance

- **[Refactor] Updated num_envs parameter to num_workers for consistency** ([#3354](https://github.com/pytorch/rl/pull/3354)) @ParamThakkar123
  - Renames `num_envs` to `num_workers` for consistency across environment wrappers

- **[Quality] replace reduce to reduction, better error message for invalid mask** ([#3179](https://github.com/pytorch/rl/pull/3179)) @Kayzwer

### Refactors (ghstack commits without PR numbers)

- **[Refactor] Move WEIGHT_SYNC_TIMEOUT to collectors._constants** (3ceb6b9a) @vmoens
  - Centralizes weight sync timeout configuration

- **[Refactor] Remove *TensorSpec classes** (7ea8d861) @vmoens
  - Removes legacy TensorSpec class aliases

- **[Refactor] Rename collectors** (8779f2a3) @vmoens
  - Renames collector classes for consistency

- **[Refactor] Non-daemonic processes in PEnv** (aeb2e9b2) @vmoens
  - Changes ParallelEnv to use non-daemonic processes

- **[Refactor] Make env creator optional for Ray** (b599d9b8) @vmoens
  - Makes environment creator optional in Ray collectors

- **[Refactor] Move decorate_thread_sub_func to torchrl.testing.mp_helpers** (e3e9e6a7) @vmoens
  - Moves multiprocessing test helpers to proper location

- **[Refactor] Use non_blocking transfers in distribution modules** (5c75777f) @vmoens
  - Uses non-blocking GPU transfers in distribution modules for better performance

- **[Refactor] Remove MUJOCO_EGL_DEVICE_ID auto-setting from torchrl init** (8fac4c54) @vmoens

- **[Refactor] Add WEIGHT_SYNC_TIMEOUT constant for collector weight synchronization** (ab3768aa) @vmoens

- **[BugFix,Test,Refactor] Refactor tests** (eb8a885b) @vmoens

- **[Refactor,Test] Move compile test to dedicated folder** (253b8ddd) @vmoens

---

## Dependencies

- **[Dependencies] Bump ray from 2.46.0 to 2.52.1** ([#3258](https://github.com/pytorch/rl/pull/3258)) @dependabot

- **[Environment] Fix envpool wrapper** ([#3339](https://github.com/pytorch/rl/pull/3339)) @vmoens
  - Updates envpool wrapper for compatibility with latest envpool versions

---

## Typing

- **[Typing] Edit wrongfully set str type annotations** (e7583b3b) @vmoens
  - Fixes incorrect string type annotations across the codebase

---

## Other

- **Revert "replace reduce to reduction, better error message for invalid mask"** ([#3182](https://github.com/pytorch/rl/pull/3182)) @vmoens

- **Fix** ([#3180](https://github.com/pytorch/rl/pull/3180)) @matteobettini

- **[Versioning] Fix doc versioning** ([#3175](https://github.com/pytorch/rl/pull/3175)) @vmoens

---

## Contributors

Thanks to all contributors to this release:

- @vmoens (Vincent Moens)
- @ParamThakkar123 (Param Thakkar)
- @bsprenger (Ben Sprenger)
- @Xmaster6y (Yoann Poupart)
- @BY571
- @LCarmi (Luca Carminati)
- @louisfaury (Faury Louis)
- @matteobettini (Matteo Bettini)
- @lowdy1
- @michalgregor (Michal Gregor)
- @cmdout
- @sebimarkgraf (Sebastian Moßburger)
- @Kayzwer
- @marcosgalleterobbva (Marcos Galletero Romero)
- @dependabot
