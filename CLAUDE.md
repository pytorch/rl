# Contributing rules for AI agents (Claude, Codex, …)

These are the house rules for any LLM-driven contribution to TorchRL. They sit on
top of [`CONTRIBUTING.md`](CONTRIBUTING.md), which still applies. When the two
disagree, this file wins for AI-generated changes.

Read this end-to-end before editing anything.

## 1. Imports

- **No local (function/method-level) imports.** Module-top imports only.
- Two exceptions, and only two:
  - **Optional dependencies.** Gate them with a module-level
    `_has_<name> = importlib.util.find_spec("<name>") is not None`, then import
    the lib lazily inside the function — or, preferred, cache it on
    `self._<name>` the first time it is needed so subsequent calls don't re-run
    `import`.
  - **Genuine circular imports.** Before deferring, try
    `from typing import TYPE_CHECKING` with a guarded import — that handles the
    type-annotation case without paying a runtime cost.
- **No wildcard imports** (`from x import *`). They have bitten us; remove them
  if you see them.
- **`from __future__ import annotations`** at the top of every new `.py` file.

## 2. Cross-version compatibility

- Use `implement_for` (re-exported as `torchrl.implement_for`, originally from
  `pyvers`) to dispatch on dependency versions — torch, gym, gymnasium, etc.
  Do not hand-roll `if torch.__version__ >= …` branches.

## 3. Logging, printing, timing

- **Never use `print()`** in library code. Use the logger:
  ```python
  from torchrl._utils import logger as torchrl_logger
  torchrl_logger.info("…")
  ```
  (also re-exported as `torchrl.torchrl_logger`).
- **For timing**, use `torchrl.timeit` — never ad-hoc `time.time()` blocks.

## 4. TensorDict-first

- New modules, transforms, losses, collectors, and replay-buffer components
  should accept and return `TensorDict` / `TensorDictBase`. Do not invent
  parallel dict-like containers.
- New objectives expose their tensordict keys via the `_AcceptedKeys`
  dataclass and the `set_keys()` plumbing used by the existing losses. Follow
  that pattern; don't bypass it.

## 5. `torch.compile` / cudagraphs friendliness

Not mandatory, **strongly encouraged**. In practice:

- Prefer `torch.where(...)` and masking over Python-level `if`/`else` on tensor
  values.
- Avoid data-dependent shapes and `.item()` calls on hot paths.
- Keep tensor dtypes/devices stable across calls.

If a component is on a hot path (collectors, replay buffers, losses, key
transforms), please verify it under `torch.compile` and, where reasonable,
under cudagraphs.

## 6. Tests

- **Every new public class / function needs tests.**
- **Do not create new test files** when an existing one already covers the
  module/area — extend the existing file. Obvious exceptions: a brand-new
  objective gets its own file under `test/` (e.g. `test/test_<algo>.py`),
  matching the pattern of the others.
- **New algorithms must be tested in the sota-implementations CI**, in
  addition to unit tests.

## 7. Documentation

- **Every new public class / function must be referenced** in the appropriate
  `docs/source/reference/*.rst` page. PRs that add a class but skip the `.rst`
  entry will be sent back.
- **Docstrings**: Sphinx-style (`Args:`, `Returns:`, `Examples:`) with a
  runnable `>>> …` example for every new public class.
- **Paper references**: if a new feature/algorithm is inspired by a paper,
  include the arXiv link (and a short citation) in the class docstring.
- **No emojis** anywhere — code, docstrings, comments, commit messages, PR
  bodies.

## 8. Tutorials

New "headline" features (a new algorithm family, a new collector, a new env
wrapper class, etc.) should ship a tutorial under `tutorials/`, or extend an
existing one. Take inspiration from the tutorials already in the repo. Tutorials
are **Sphinx-first**, not script-first:

- Use `# regular prose comments` for explanation, **not** `print("…")`.
- Structure should mirror existing tutos, including (names can be rephrased):
  - a **"What you will learn"** section near the top,
  - a **"Conclusion"** section,
  - a **"Further reading"** section.

## 9. Benchmarks

If a change is performance-relevant — anything on a hot path
(collectors, replay buffers, losses, transforms, env stepping) — add or extend
a benchmark under `benchmarks/`. "Performance-relevant" is the trigger; pure
correctness fixes don't need one.

## 10. SOTA implementations

A new algorithm needs:

- a runnable script in `sota-implementations/<algo>/` with a Hydra config,
- an entry in `sota-check/` so it's exercised by the SOTA CI.

## 11. Type hints

- New public signatures should carry type hints. Internal helpers can skip
  them, but prefer to add them when convenient.
- Hints are documentary (we don't enforce them with mypy in CI), but they
  must be **accurate** — wrong hints are worse than no hints.

## 12. Backwards compatibility & deprecations

We give users **two minor releases** of warning before any breaking change.
Concretely, if the next release is `0.X`:

- Deprecate in `0.X`,
- Default-value changes (if any) can land in `0.(X+1)`,
- Final removal / behavior switch in `0.(X+2)`.

Rules:

- Use `DeprecationWarning` for API removals; `FutureWarning` for upcoming
  default-value changes that users can already see.
- **Always state the schedule explicitly** in the warning message, naming the
  version where the change will happen, e.g.:
  > `MyClass.foo` is deprecated and will be removed in v0.X+2. Use
  > `MyClass.bar` instead.

## 13. PR labels

Use a `[Tag]` prefix on the PR title. Canonical set seen in the repo:

```
[Algorithm] [BE] [BugFix] [CI] [Deprecation] [Doc]
[Feature] [Minor] [Performance] [Quality] [Refactor]
[Test] [Versioning]
```

Pick the most specific one. `[Feature]` for new user-facing capability,
`[BugFix]` for fixes, `[Doc]` for docs-only, `[CI]` for workflows,
`[BE]` for backend / internal plumbing, `[Performance]` for perf work,
`[Quality]` for lint/typing/cleanup, `[Refactor]` for behavior-preserving
restructure, `[Algorithm]` when adding/modifying a learning algorithm,
`[Deprecation]` for deprecation warnings, `[Versioning]` for version bumps.

## 14. Commits

No squash requirement. Make commits that read sensibly on their own; that's
all.

## 15. When in doubt

Read a recently-merged PR in the same area and match the conventions there
before inventing your own.
