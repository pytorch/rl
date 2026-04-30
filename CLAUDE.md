# Contributing rules for AI agents

House rules for LLM contributions to TorchRL. Sits on top of `CONTRIBUTING.md`;
this file wins on conflicts. Read end-to-end before editing.

## 1. Imports

- Module-top imports only. No function/method-level imports. Two exceptions:
  - **Optional deps**: `_has_<name> = importlib.util.find_spec("<name>") is not None`
    at module top, then lazy import (preferably cached on `self._<name>`).
  - **Genuine circular imports**: try `from typing import TYPE_CHECKING` first.
- No wildcard imports (`from x import *`).
- `from __future__ import annotations` at the top of every new `.py` file.

## 2. Cross-version compatibility

Use `torchrl.implement_for` (from `pyvers`) for version dispatch on torch / gym /
gymnasium / etc. No hand-rolled `if torch.__version__ >= …` branches.

## 3. Logging, printing, timing

- No `print()` in library code. Use `from torchrl._utils import logger as torchrl_logger`
  (also `torchrl.torchrl_logger`).
- Timing: `torchrl.timeit`, never `time.time()` blocks.

## 4. TensorDict-first

- New modules / transforms / losses / collectors / RB components accept and
  return `TensorDict` / `TensorDictBase`. No parallel dict-like containers.
- New objectives expose tensordict keys via `_AcceptedKeys` + `set_keys()`,
  matching existing losses.

## 5. Type hints & annotations

- Public signatures carry type hints. Hints must be accurate (not enforced by
  mypy, but wrong hints are worse than none).
- **Prefer `NestedKey` over `str`** for tensordict keys, unless the value
  genuinely cannot be a `tensordict.NestedKey`.
- **Use `Literal[...]`** for any fixed set of string values (e.g.
  `mode: Literal["random", "greedy"]`), not bare `str`.

## 6. `torch.compile` / cudagraphs friendliness

Strongly encouraged (not mandatory):

- Prefer `torch.where(...)` / masking over Python `if`/`else` on tensor values.
- Avoid data-dependent shapes and `.item()` on hot paths.
- Keep dtypes/devices stable across calls.
- Hot-path components (collectors, RB, losses, key transforms): verify under
  `torch.compile` and, where reasonable, cudagraphs.

## 7. Tests

- Every new public class / function needs tests.
- **Do not create new test files** when an existing one covers the area —
  extend it. Exception: a brand-new objective gets `test/test_<algo>.py`.
- If your module accepts a `NestedKey` input, add a test exercising a nested
  key (not just a flat string).
- New algorithms: also tested in the sota-implementations CI.

## 8. Documentation

- Every new public class / function referenced in `docs/source/reference/*.rst`.
- Sphinx-style docstrings (`Args:`, `Returns:`, `Examples:`) with a runnable
  `>>> …` example for every new public class.
- Paper references: include the arXiv link + short citation in the class
  docstring.
- No emojis anywhere — code, docstrings, comments, commits, PR bodies.

## 9. Tutorials

New "headline" features (algorithm family, collector, env wrapper) ship a
tutorial under `tutorials/` (or extend an existing one). Sphinx-first:

- `# prose comments` for explanation, **not** `print(...)`.
- Include "What you will learn", "Conclusion", "Further reading" sections
  (names can be rephrased), mirroring existing tutos.

## 10. Benchmarks

Performance-relevant changes (anything on a hot path: collectors, RB, losses,
transforms, env stepping) add/extend a benchmark under `benchmarks/`. Pure
correctness fixes don't need one.

## 11. SOTA implementations

New algorithm needs: a runnable script under `sota-implementations/<algo>/`
with a Hydra config, plus an entry in `sota-check/`.

## 12. Backwards compatibility & deprecations

Two minor releases of warning before any breaking change. If next release is
`0.X`: deprecate in `0.X`, default-value changes in `0.(X+1)`, final removal
in `0.(X+2)`.

- `DeprecationWarning` for API removals; `FutureWarning` for upcoming default
  changes.
- **Always name the target version explicitly** in the warning, e.g.
  `"MyClass.foo is deprecated and will be removed in v0.X+2. Use MyClass.bar."`

## 13. PR labels & commits

`[Tag]` prefix on PR title. Canonical set:

```
[Algorithm] [BE] [BugFix] [CI] [Deprecation] [Doc]
[Feature] [Minor] [Performance] [Quality] [Refactor]
[Test] [Versioning]
```

Pick the most specific. No squash requirement on commits — just make each
commit read sensibly on its own.

## 14. Config / class parity

Some classes (Trainers, losses, RB components, transforms, …) have a Hydra
`*Config` dataclass companion under `torchrl/trainers/algorithms/configs/`.

- **Parity.** Every kwarg of the wrapped class's `__init__` must appear as a
  Config field (same default), be popped in the matching `_make_*` factory,
  and forwarded to the constructor. Adding a kwarg without surfacing it in the
  Config silently breaks Hydra users.
- **Cross-references.** Config docstring references its class
  (`Hydra configuration for :class:`~torchrl.trainers.algorithms.SACTrainer``);
  class docstring references the Config
  (`See also :class:`~torchrl.trainers.algorithms.configs.SACTrainerConfig``).
- **When in doubt**:
  `git grep -n "class .*Config(" torchrl/trainers/algorithms/configs/` and
  match the existing pattern.

## 15. When in doubt

Read a recently-merged PR in the same area and match its conventions.
