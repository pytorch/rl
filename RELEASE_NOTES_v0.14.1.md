# TorchRL v0.14.1

TorchRL 0.14.1 is a maintenance release covering environment resets, collectors, replay buffers, planning, training, logging, and LLM workflows. It also corrects transform device handling and introduces the v0.17 device deprecations described below.

# Heads-up: an upcoming release will require torch>=2.13

An upcoming TorchRL release will require `torch>=2.13` (discussed in [#4523](https://github.com/pytorch/rl/discussions/4523)). The existing lower bound on the `torch` version is `2.1.0`, so TorchRL can be installed next to torch releases that lack APIs it calls. Source builds and NGC containers of torch 2.13, which report versions such as `2.13.0a0+<sha>`, will also meet the requirement.

TorchRL 0.14.1 and earlier releases keep working with older torch. Once the requirement is released:
- If your environment pins torch below 2.13, pip and uv will install the newest TorchRL release that accepts your pinned torch version.
- If your environment has an older torch but does not pin it, upgrading TorchRL will also upgrade torch. To keep your current torch, pin it.

## Compatibility and installation

Release builds are configured for the latest stable PyTorch, **2.14.1**. TorchRL 0.14.1 requires **TensorDict >=0.14.3,<0.15.0**.

Install the release with:

```bash
pip install "torch==2.14.1" "torchrl==0.14.1" "tensordict>=0.14.3,<0.15.0"
```

## Bug fixes

### Environments and datasets

- Wrapping an existing `TransformedEnv` without an additional transform now preserves its inner transforms and constructs successfully. [#4336](https://github.com/pytorch/rl/pull/4336) by @vmoens.
- `ParallelEnv(use_buffers=False)` preserves the observations and done flags of workers left untouched by a partial reset. Consolidation also avoids locking the caller's reset data. [#4339](https://github.com/pytorch/rl/pull/4339) by @vmoens.
- `RandomTruncationTransform` reads nested step counters and writes the corresponding nested truncation and done flags. Optional `step_count_key`, `truncated_key`, and `done_key` arguments support customized keys, and partial resets preserve other workers' horizons. [#4350](https://github.com/pytorch/rl/pull/4350) by @YeonwooSung.
- `TicTacToeEnv` computes wins independently across batched boards, handles mixed finished and active single-player games, and accepts `rand_action(None)`. See the return-type migration note below for direct calls to `win`. [#4349](https://github.com/pytorch/rl/pull/4349) by @YeonwooSung.
- `GymEnv` imports `ale_py` before resolving Atari `NoFrameskip` IDs and `ale_py:`-prefixed IDs, as well as the existing `ALE/` namespace. This fixes environment registration where plugins are not loaded automatically. [#4358](https://github.com/pytorch/rl/pull/4358) by @YeonwooSung.
- Minari dataset conversion allocates separate non-tensor storage for current and next observations. Mission text now stays aligned with each transition instead of being overwritten through shared storage. [#4256](https://github.com/pytorch/rl/pull/4256) by @YeonwooSung.
- `SerialEnv` and `ParallelEnv` preserve worker batch dimensions when partial-step masks select whole workers. A mask selecting only part of a worker now raises `ValueError`; place the partial-step transform inside each worker for that case. `BatchSubSampler` treats leading batch dimensions as independent trajectories. [#4389](https://github.com/pytorch/rl/pull/4389) by @vmoens.
- `ActionDiscretizer` constructs exactly the requested number of bins, including for counts where floating-point step accumulation previously produced an extra bin. [#4443](https://github.com/pytorch/rl/pull/4443) by @Nicholas022400701.
- `Choice.squeeze(dim)` also squeezes its constituent choices, keeping their shapes consistent with the returned spec. [#4462](https://github.com/pytorch/rl/pull/4462) by @yupengtang.
- Integer `Bounded.rand` can sample its inclusive upper bound and avoids overflow when bounds span a wide range. [#4497](https://github.com/pytorch/rl/pull/4497) by @jayzuccarelli and [#4498](https://github.com/pytorch/rl/pull/4498) by @theap06.
- `MultiCategorical.to_one_hot` handles batched specs and scalar samples from a shape-`[1]` spec. [#4504](https://github.com/pytorch/rl/pull/4504) by @Nicholas022400701.
- Isaac Lab integration supports Isaac Lab 3.x tensor conversion and partial resets in `DirectRLEnvWarp`. [#4194](https://github.com/pytorch/rl/pull/4194) by @theap06.
- `ParallelEnv` shutdown errors now name the closed environment class, and `DiscreteCQLLoss` missing-key errors include the actual Q-value key. [#4496](https://github.com/pytorch/rl/pull/4496) by @David-Wu1119.

### Replay buffers, planning, and training

- `TensorDictMaxValueWriter` restores checkpointed heap entries with the types required for subsequent writes. Restored replay buffers can continue adding data and replacing their lowest-ranked entries without comparison errors. [#4370](https://github.com/pytorch/rl/pull/4370) by @yupengtang.
- `CEMPlanner` and `MPPIPlanner` retain the full planning horizon while excluding rewards after the first termination or truncation from trajectory scores. The reward on the done step is retained, and nested done groups follow environment precedence. [#4359](https://github.com/pytorch/rl/pull/4359) by @YeonwooSung.
- `SACLossConfig` forwards only the fields accepted by the selected continuous or discrete SAC loss, preventing constructor errors from variant-specific arguments. [#4338](https://github.com/pytorch/rl/pull/4338) by @yupengtang.
- `CrossQLoss.set_keys` works after a value estimator has been created. Renamed reward, done, and terminated keys propagate to the estimator without accessing a nonexistent value key. [#4372](https://github.com/pytorch/rl/pull/4372) by @yupengtang.
- TD3 trainer construction works with multiprocessing collectors that do not expose an `env` attribute when the loss partial supplies `action_spec` or `bounds`. Explicit action domains are preserved; collectors with an environment still provide inferred bounds when needed. [#4368](https://github.com/pytorch/rl/pull/4368) by @aswanth-07.
- Shared replay schemas retain policy-version fields, including next-step versions, when collectors initialize replay storage. [#4407](https://github.com/pytorch/rl/pull/4407) by @vmoens.
- Sampler state survives repeated replay-buffer checkpoints, so a second save retains the current sampling state. [#4388](https://github.com/pytorch/rl/pull/4388) by @vmoens.
- `ReplayBuffer.can_sample()` holds the replay and write locks while checking sampler readiness, avoiding races with concurrent writers. [#4439](https://github.com/pytorch/rl/pull/4439) by @vmoens.
- `ParameterScheduler` honors explicit zero minimum and maximum values and omits its backend module from `state_dict`, making saved state serializable. [#4444](https://github.com/pytorch/rl/pull/4444) and [#4445](https://github.com/pytorch/rl/pull/4445) by @Nicholas022400701.
- SAC and CQL trainers query collector action specs only when automatic target entropy needs one and the actor does not already supply a spec. [#4377](https://github.com/pytorch/rl/pull/4377) by @aswanth-07.
- Hydra can instantiate sampler and storage ensemble configs, including their nested components. [#4483](https://github.com/pytorch/rl/pull/4483) by @aswanth-07.
- `CatTensorsConfig` uses the configured concatenation key, and `InitTrackerConfig` defaults to the same `is_init` key as `InitTracker`. [#4426](https://github.com/pytorch/rl/pull/4426) and [#4428](https://github.com/pytorch/rl/pull/4428) by @bsprenger.
- `GRPOLoss` handles reference log-probability padding and custom reference keys correctly, while DAPO loss construction no longer fails on an invalid initializer. [#4363](https://github.com/pytorch/rl/pull/4363) by @coder-jayp.
- `GAEConfig.average_gae` now matches the `GAE` constructor default. See the migration note below if you relied on the previous config default. [#4472](https://github.com/pytorch/rl/pull/4472) by @YeonwooSung.
- `ReinforceLoss` scores actions stored in the input tensordict under the actor's current distribution, including composite actions, instead of sampling replacement actions. [#4525](https://github.com/pytorch/rl/pull/4525) by @theap06.
- Trainer scalar logging computes standard deviations for categorical integer values without a dtype error. [#4390](https://github.com/pytorch/rl/pull/4390) by @vmoens.
- DreamerV3 Triton recurrent matmul avoids an unreliable B2/K16 autotuning tile in forward and backward passes. [#4257](https://github.com/pytorch/rl/pull/4257) by @YeonwooSung.

### Collectors and distributed execution

- Multiprocess collector shutdown is bounded and repeatable, avoiding warnings for expected worker disconnects. Spawn-bootstrap warnings are also filtered at their source. [#4409](https://github.com/pytorch/rl/pull/4409) and [#4411](https://github.com/pytorch/rl/pull/4411) by @vmoens.
- Delayed distributed collectors forward policy factories and weight-sync settings to workers, preserving their configured behavior. [#4447](https://github.com/pytorch/rl/pull/4447) by @aswanth-07.
- `MultiAsyncCollector` checks worker liveness after every batch and honors `update_at_each_batch` during policy updates. [#4505](https://github.com/pytorch/rl/pull/4505) by @theap06.
- `MultiSyncCollector` and `MultiAsyncCollector` assign fresh trajectory IDs on iteration resets while partial resets keep unaffected trajectories. [#4515](https://github.com/pytorch/rl/pull/4515) by @theap06.
- `RayCollector` releases completed object references without calling the removed `ray.internal.free` API. [#4519](https://github.com/pytorch/rl/pull/4519) by @coder-jayp.

### Logging and resume

- `WandbLogger` creates its save directory before initializing a run, so logging can start when that directory does not yet exist. [#4405](https://github.com/pytorch/rl/pull/4405) by @vmoens.
- `CSVLogger` counts text and video steps independently. Loading legacy checkpoints also preserves existing text and hyperparameter files by continuing their numbering instead of overwriting them. [#4374](https://github.com/pytorch/rl/pull/4374) by @yupengtang.
- Logger checkpoints preserve `log_dir=None` and store configured directories as absolute paths, so resuming after a working-directory change uses the original location. `WandbLogger` rejects loading state from a different live run instead of silently changing its recorded identity. Logger fixes from [#4382](https://github.com/pytorch/rl/pull/4382) by @vmoens.

### LLM workflows and transforms

- `vLLMWrapper` no longer fills missing prompt log-probabilities with invented zero scores or exposes response-only scores as a full sequence. Full-sequence alignment and GRPO/KL masking remain correct when vLLM generation omits prompt scores. [#4255](https://github.com/pytorch/rl/pull/4255) by @YeonwooSung.
- `LLMCollector(yield_only_last_steps=True)` counts intermediate dialog turns toward `total_dialog_turns`, even though it returns only final steps. Asynchronous prefetch also accounts for pending and in-progress turns when deciding whether to launch more work. [#4354](https://github.com/pytorch/rl/pull/4354) by @YeonwooSung.
- `ChatEnv.reset` preserves roles and conversation structure when its query contains a `History` or chat-message list. Existing conversations, including batched non-tensor wrappers, receive the configured system prompt correctly. [#4355](https://github.com/pytorch/rl/pull/4355) by @YeonwooSung.
- `IncrementalTokenizer` derives the reusable full-token key from the complete configured `NestedKey`, preserving deep prefixes and handling string and single-component keys consistently. [#4351](https://github.com/pytorch/rl/pull/4351) by @YeonwooSung.
- `Tokenizer` follows its parent environment's current device for token tensors and attention masks instead of caching an earlier device. The output destination is exposed through `out_device`. [#4352](https://github.com/pytorch/rl/pull/4352) by @YeonwooSung.
- `ModuleTransform`, `KLRewardTransform`, and `RetrieveLogProb` place wrapped models on the constructor device and update an explicit input/output placement policy when `.to(device)` is called. `KLRewardTransform` registers its reference model as a submodule so device moves reach it. [#4353](https://github.com/pytorch/rl/pull/4353) by @YeonwooSung.
- `BrowserTransform` starts a usable event loop when called from a thread without one. [#4485](https://github.com/pytorch/rl/pull/4485) by @David-Wu1119.

## Migration and deprecations

- `GAEConfig.average_gae` now defaults to `False` instead of `True`, matching direct `GAE` construction. Set `average_gae=True` explicitly to retain the previous Hydra-configured behavior. [#4472](https://github.com/pytorch/rl/pull/4472) by @YeonwooSung.
- `InitTrackerConfig.init_key` now defaults to `"is_init"` instead of `None`, matching `InitTracker`. Set `init_key=None` explicitly if your configuration relied on the old value. [#4428](https://github.com/pytorch/rl/pull/4428) by @bsprenger.
- `WandbLogger.load_state_dict` raises `RuntimeError` when the saved run ID differs from the active run ID. Construct the logger with the saved ID and `resume="must"` before loading, or use `get_logger(..., state_dict=saved_state)` to resume the saved run. [#4382](https://github.com/pytorch/rl/pull/4382) by @vmoens.
- When vLLM generation omits usable prompt scores, `LogProbs.prompt` remains unset (`None`) and `LogProbs.full` contains `NaN` at unavailable prompt positions while retaining prompt-plus-response length. Custom consumers should mask these alignment entries; `GRPOLoss` and `KLComputation` exclude unavailable prompt positions from their computations. Use `generate=False` when prompt scores are required. [#4255](https://github.com/pytorch/rl/pull/4255) by @YeonwooSung.
- `TicTacToeEnv.win` now returns a boolean tensor with shape `[..., 1]` instead of a Python boolean. Direct callers should handle the batch-shaped result; use `.item()` only when a scalar result is required. [#4349](https://github.com/pytorch/rl/pull/4349) by @YeonwooSung.
- `Tokenizer.device` is deprecated and will be removed in **v0.17**. Use `Tokenizer.out_device`, which follows the parent environment; `None` leaves outputs on the tokenizer's chosen device. [#4352](https://github.com/pytorch/rl/pull/4352) by @YeonwooSung.
- The `device` attributes of `ModuleTransform`, `KLRewardTransform`, and `RetrieveLogProb` are deprecated and will be removed in **v0.17**. They describe an explicit input/output placement policy, not every parameter or buffer's location. Place models with the constructor `device=` argument or `.to(device)`, inspect individual tensors when checking placement, and let wrapped modules own device routing. Until v0.17, constructor `device=` retains the I/O policy and later `.to(device)` updates it; dtype-only moves leave it unchanged. [#4353](https://github.com/pytorch/rl/pull/4353) by @YeonwooSung.

## Release infrastructure, documentation, and validation

- Release wheel builds select and verify stable PyTorch through test-infra `release/2.14`; the package version files are set to 0.14.1. [d844b7b8f](https://github.com/pytorch/rl/commit/d844b7b8f1b09030a76325389d936b4375905820) by @vmoens.
- Multi-agent environment documentation now explains group-nested tensordict layouts, the return contract for custom `_step()` methods, and action, reward, and done keys across multiple groups, with a two-team example. [#4232](https://github.com/pytorch/rl/pull/4232) by @YeonwooSung.
- Documentation deployment now copies built pages through the `linux_job_v3` job. [#4524](https://github.com/pytorch/rl/pull/4524) by @huydhn.
- Linux x86 and aarch64 wheel jobs use the OSDC runner fleet; container setup scripts were updated for those runners. [#4487](https://github.com/pytorch/rl/pull/4487) and [#4531](https://github.com/pytorch/rl/pull/4531) by @huydhn.
- GPU unit tests cap parallel workers and clean up stray processes at job exit. [#4532](https://github.com/pytorch/rl/pull/4532) by @huydhn.
- `NoisyLinear`, `NoisyLazyLinear`, and `reset_noise` documentation clarifies that forward passes reuse existing noise. Call `module.apply(reset_noise)` when a fresh noise sample is needed, including after optimization steps. [#4357](https://github.com/pytorch/rl/pull/4357) by @YeonwooSung.
- Prioritized slice-sampler tests allow the small nonzero sampling probability assigned to zero-priority episodes by epsilon, while still checking that preferred episodes dominate. This changes validation only. [#4364](https://github.com/pytorch/rl/pull/4364) by @vmoens.

## Contributors

Thanks to @aswanth-07, @bsprenger, @coder-jayp, @David-Wu1119, @huydhn, @jayzuccarelli, @Nicholas022400701, @theap06, @vmoens, @YeonwooSung, and @yupengtang for the changes included in this release.

[Full changelog: v0.14.0...v0.14.1](https://github.com/pytorch/rl/compare/v0.14.0...v0.14.1)
