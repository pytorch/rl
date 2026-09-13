# TorchRL v0.14.1

TorchRL 0.14.1 is a maintenance release with 14 bug fixes covering environment resets, planning, trainer construction, and LLM workflows. It also corrects transform device handling and introduces the v0.17 device deprecations described below.

## Compatibility and installation

Release builds are configured for the latest stable PyTorch, **2.14.0**. TorchRL 0.14.1 requires **TensorDict >=0.14.2,<0.15.0**.

After publication, install the release with:

```bash
pip install "torch==2.14.0" "torchrl==0.14.1" "tensordict>=0.14.2,<0.15.0"
```

## Bug fixes

### Environments and datasets

- Wrapping an existing `TransformedEnv` without an additional transform now preserves its inner transforms and constructs successfully. [#4336](https://github.com/pytorch/rl/pull/4336) by @vmoens.
- `ParallelEnv(use_buffers=False)` preserves the observations and done flags of workers left untouched by a partial reset. Consolidation also avoids locking the caller's reset data. [#4339](https://github.com/pytorch/rl/pull/4339) by @vmoens.
- `RandomTruncationTransform` reads nested step counters and writes the corresponding nested truncation and done flags. Optional `step_count_key`, `truncated_key`, and `done_key` arguments support customized keys, and partial resets preserve other workers' horizons. [#4350](https://github.com/pytorch/rl/pull/4350) by @YeonwooSung.
- `TicTacToeEnv` computes wins independently across batched boards, handles mixed finished and active single-player games, and accepts `rand_action(None)`. See the return-type migration note below for direct calls to `win`. [#4349](https://github.com/pytorch/rl/pull/4349) by @YeonwooSung.
- `GymEnv` imports `ale_py` before resolving Atari `NoFrameskip` IDs and `ale_py:`-prefixed IDs, as well as the existing `ALE/` namespace. This fixes environment registration where plugins are not loaded automatically. [#4358](https://github.com/pytorch/rl/pull/4358) by @YeonwooSung.
- Minari dataset conversion allocates separate non-tensor storage for current and next observations. Mission text now stays aligned with each transition instead of being overwritten through shared storage. [#4256](https://github.com/pytorch/rl/pull/4256) by @YeonwooSung.

### Planning and trainers

- `CEMPlanner` and `MPPIPlanner` retain the full planning horizon while excluding rewards after the first termination or truncation from trajectory scores. The reward on the done step is retained, and nested done groups follow environment precedence. [#4359](https://github.com/pytorch/rl/pull/4359) by @YeonwooSung.
- `SACLossConfig` forwards only the fields accepted by the selected continuous or discrete SAC loss, preventing constructor errors from variant-specific arguments. [#4338](https://github.com/pytorch/rl/pull/4338) by @yupengtang.
- TD3 trainer construction works with multiprocessing collectors that do not expose an `env` attribute when the loss partial supplies `action_spec` or `bounds`. Explicit action domains are preserved; collectors with an environment still provide inferred bounds when needed. [#4368](https://github.com/pytorch/rl/pull/4368) by @aswanth-07.

### LLM workflows and transforms

- `LLMCollector(yield_only_last_steps=True)` counts intermediate dialog turns toward `total_dialog_turns`, even though it returns only final steps. Asynchronous prefetch also accounts for pending and in-progress turns when deciding whether to launch more work. [#4354](https://github.com/pytorch/rl/pull/4354) by @YeonwooSung.
- `ChatEnv.reset` preserves roles and conversation structure when its query contains a `History` or chat-message list. Existing conversations, including batched non-tensor wrappers, receive the configured system prompt correctly. [#4355](https://github.com/pytorch/rl/pull/4355) by @YeonwooSung.
- `IncrementalTokenizer` derives the reusable full-token key from the complete configured `NestedKey`, preserving deep prefixes and handling string and single-component keys consistently. [#4351](https://github.com/pytorch/rl/pull/4351) by @YeonwooSung.
- Both `Tokenizer` transforms follow their parent environment's current device for token tensors and attention masks instead of caching an earlier device. The output destination is exposed through `out_device`. [#4352](https://github.com/pytorch/rl/pull/4352) by @YeonwooSung.
- `ModuleTransform`, `KLRewardTransform`, and `RetrieveLogProb` place wrapped models on the constructor device and update an explicit input/output placement policy when `.to(device)` is called. `KLRewardTransform` registers its reference model as a submodule so device moves reach it. [#4353](https://github.com/pytorch/rl/pull/4353) by @YeonwooSung.

## Migration and deprecations

- `TicTacToeEnv.win` now returns a boolean tensor with shape `[..., 1]` instead of a Python boolean. Direct callers should handle the batch-shaped result; use `.item()` only when a scalar result is required. [#4349](https://github.com/pytorch/rl/pull/4349) by @YeonwooSung.
- `Tokenizer.device` is deprecated and will be removed in **v0.17**. Use `Tokenizer.out_device`, which follows the parent environment; `None` leaves outputs on the tokenizer's chosen device. [#4352](https://github.com/pytorch/rl/pull/4352) by @YeonwooSung.
- The `device` attributes of `ModuleTransform`, `KLRewardTransform`, and `RetrieveLogProb` are deprecated and will be removed in **v0.17**. They describe an explicit input/output placement policy, not every parameter or buffer's location. Place models with the constructor `device=` argument or `.to(device)`, inspect individual tensors when checking placement, and let wrapped modules own device routing. Until v0.17, constructor `device=` retains the I/O policy and later `.to(device)` updates it; dtype-only moves leave it unchanged. [#4353](https://github.com/pytorch/rl/pull/4353) by @YeonwooSung.

## Documentation and validation

- `NoisyLinear`, `NoisyLazyLinear`, and `reset_noise` documentation clarifies that forward passes reuse existing noise. Call `module.apply(reset_noise)` when a fresh noise sample is needed, including after optimization steps. [#4357](https://github.com/pytorch/rl/pull/4357) by @YeonwooSung.
- Prioritized slice-sampler tests allow the small nonzero sampling probability assigned to zero-priority episodes by epsilon, while still checking that preferred episodes dominate. This changes validation only. [#4364](https://github.com/pytorch/rl/pull/4364) by @vmoens.

## Contributors

Thanks to @aswanth-07, @vmoens, @YeonwooSung, and @yupengtang for the changes included in this release.

[Full changelog: v0.14.0...v0.14.1](https://github.com/pytorch/rl/compare/v0.14.0...v0.14.1)
