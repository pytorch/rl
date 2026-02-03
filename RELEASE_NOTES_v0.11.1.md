# TorchRL v0.11.1 Release Notes

## Highlights

This patch release fixes two bugs related to TensorDict batch sizes and multi-agent environment handling:

- Fixed `Composite.encode()` to correctly set the batch size of the output TensorDict
- Fixed `StepCounter` to properly track nested truncated and done states in multi-agent environments

## Breaking Changes

No breaking changes in this release.

## Bug Fixes

- **Fixed batch size in `Composite.encode`**: The `Composite.encode()` method now correctly sets the `batch_size` of the output `TensorDict` to match the shape of the tensor spec, rather than returning an empty batch size. ([#3411](https://github.com/pytorch/rl/pull/3411)) - @tobiabir
  
  Previously, calling `Composite.encode(raw_vals)` would return a TensorDict with `batch_size=torch.Size([])` regardless of the spec's shape. This is now fixed to return the correct batch size matching the spec shape.

- **Fixed `StepCounter` nested done/truncated tracking in multi-agent environments**: `StepCounter` now properly updates nested truncated and done keys for multi-agent environments. ([#3405](https://github.com/pytorch/rl/pull/3405)) - @vmoens
  
  When using `StepCounter` with multi-agent environments (e.g., PettingZoo), the transform now correctly propagates truncated/done signals to agent-specific keys (e.g., `("agent", "truncated")`) in addition to the root-level keys. This ensures consistent episode termination tracking across all agent groups.

## Internal / CI Improvements

These changes are internal and do not affect the public API:

- Upgraded `meshgrid` usage to address PyTorch deprecation warning ([#3412](https://github.com/pytorch/rl/pull/3412)) - @vmoens
- Added flaky test tracking system for improved CI reliability ([#3408](https://github.com/pytorch/rl/pull/3408)) - @vmoens
- Added file-based auto-labeling for PR components ([#3402](https://github.com/pytorch/rl/pull/3402)) - @vmoens
- Improved LLM prompt for release workflow ([#3399](https://github.com/pytorch/rl/pull/3399)) - @vmoens

## Contributors

Thanks to all contributors to this release:

- @tobiabir (Tobias Birchler) - First-time contributor!
- @vmoens (Vincent Moens)

## Installation

```bash
pip install torchrl==0.11.1
```

Or with conda:

```bash
conda install -c pytorch torchrl=0.11.1
```
