# TorchRL v0.13.3

TorchRL 0.13.3 is a patch release focused on correctness, state restoration, device metadata, and environment reliability. It also includes the backward-compatible value-estimator chunk-dimension option from [#4003](https://github.com/pytorch/rl/pull/4003).

## Backported pull-request inventory

### Value estimation and objectives

- [#4003](https://github.com/pytorch/rl/pull/4003) by @lin-erica adds `value_chunk_dim`, allowing value estimators and their Hydra configs to chunk along a selected batch dimension while preserving the existing default.
- [#3936](https://github.com/pytorch/rl/pull/3936) by @vmoens preserves PPO effective-sample-size feature dimensions when the leading batch dimension is a singleton.
- [#3888](https://github.com/pytorch/rl/pull/3888) by @coder-jayp routes the remaining loss modules through the common mask-aware reduction path, making masking behavior consistent across objectives.
- [#3886](https://github.com/pytorch/rl/pull/3886) and [#3887](https://github.com/pytorch/rl/pull/3887) by @fallintoplace normalize reward-model losses over valid preference pairs and infer padding IDs from the selected model tokenizer, with an explicit override when needed.

### Replay buffers and collectors

- [#3871](https://github.com/pytorch/rl/pull/3871) by @Agade09 fixes the replay-buffer prefetch queue off-by-one error so the configured number of prefetched futures is maintained.
- [#3912](https://github.com/pytorch/rl/pull/3912), [#3914](https://github.com/pytorch/rl/pull/3914), and [#3915](https://github.com/pytorch/rl/pull/3915) by @vmoens normalize root and nested `TensorDict` device metadata before collector writes into replay-buffer storage, without moving incompatible tensor leaves.
- [#3925](https://github.com/pytorch/rl/pull/3925) by @theap06 fixes priorities being transformed by alpha twice in `PrioritizedSampler`.
- [#3966](https://github.com/pytorch/rl/pull/3966) by @vmoens clones incoming replay-buffer, sampler, and trainer tensors during `load_state_dict`, preventing live state from aliasing caller-owned or memory-mapped tensors.
- [#3990](https://github.com/pytorch/rl/pull/3990) by @theap06 restores persistence dispatch for `SliceSamplerWithoutReplacement` and `PrioritizedSliceSampler` despite their multiple-inheritance method resolution order.

### Environments and integrations

- [#3873](https://github.com/pytorch/rl/pull/3873) by @vmoens preserves dynamic spec dimensions and batched non-tensor state in batched environments.
- [#3867](https://github.com/pytorch/rl/pull/3867) by @discobot makes `ParallelEnv` default to pipe transport for MPS-backed environments and stages pipe data through CPU memory.
- [#3916](https://github.com/pytorch/rl/pull/3916) by @vmoens skips aliased MuJoCo-Torch leaves during masked resets, avoiding duplicate writes to shared backend state.
- [#3920](https://github.com/pytorch/rl/pull/3920) by @younik scopes `MINARI_DATASETS_PATH` around `minari.load_dataset`, allowing datasets downloaded into TorchRL's cache to load correctly.
- [#3961](https://github.com/pytorch/rl/pull/3961) by @vmoens restores an unset `AUTO_UNWRAP_TRANSFORMED_ENV` variable without leaking the literal string `"None"` into child processes.
- [#3899](https://github.com/pytorch/rl/pull/3899) by @vmoens stabilizes optional chess and Jumanji installs and decodes rendered PNGs directly with torchvision.

### Dependency and test reliability

- [#3969](https://github.com/pytorch/rl/pull/3969) by @vmoens requires `pyvers>=0.2.3`, ensuring lazy `implement_for` wrappers are pickleable before their first dispatch.
- [#3960](https://github.com/pytorch/rl/pull/3960) by @vmoens provisions a compatible Miniconda environment for macOS wheel builds after the runner base moved to Python 3.14.
- [#3970](https://github.com/pytorch/rl/pull/3970) by @vmoens makes mock-environment observation keys deterministic even when an explicit observation spec is used for the first instance in a process.
- [#3991](https://github.com/pytorch/rl/pull/3991) by @vmoens removes a seed-dependent `RewardScaling` round-trip assertion failure near zero.

## Highlights

- Value-estimator chunking can now target any valid batch dimension through `value_chunk_dim`, including the matching trainer configuration field.
- Replay-buffer sampling, state restoration, collector writes, and prioritized replay receive several correctness fixes.
- Batched, MPS, MuJoCo-Torch, Minari, chess, and Jumanji environment paths are more robust.
- The release keeps the TorchRL 0.13 dependency line and the same PyTorch 2.11 wheel matrix used for 0.13.2.

## Breaking changes

No breaking changes are intended in this release.

## Public API exports

No new package-level public symbols are exported by this patch. The additive `value_chunk_dim` constructor/configuration option is documented under [#4003](https://github.com/pytorch/rl/pull/4003).

## Contributors

Thanks to @Agade09, @coder-jayp, @discobot, @fallintoplace, @lin-erica, @theap06, @vmoens, and @younik for the changes included in this release.
