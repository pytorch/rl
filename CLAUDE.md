# Contributing rules for AI agents

House rules for LLM contributions to TorchRL. Sits on top of `CONTRIBUTING.md`;
this file wins on conflicts. Read end-to-end before editing.

The overarching rule: **stay within TorchRL's contracts.** When a task can be
expressed with an existing component (a `TensorDictModule`, `env.rollout`, a
transform, a collector, a replay buffer, a logger), use it. Do not write a
bespoke loop, container or helper that re-implements what the library already
does. A reviewer who has to read a hand-rolled stepping loop or a list of
dataclasses of floats will send the PR back.

## 1. Imports

- Module-top imports only. No function/method-level imports. Two exceptions:
  - **Optional deps**: `_has_<name> = importlib.util.find_spec("<name>") is not None`
    at module top, then lazy import (preferably cached on `self._<name>`).
  - **Genuine circular imports**: try `from typing import TYPE_CHECKING` first.
- No wildcard imports (`from x import *`).
- `from __future__ import annotations` at the top of every new `.py` file.

## 2. Cross-version compatibility

Use `torchrl.implement_for` (from `pyvers`) for version dispatch on torch / gym /
gymnasium / etc. No hand-rolled `if torch.__version__ >= ...` branches.

## 3. Logging, printing, timing

- No `print()` in library code. Use `from torchrl._utils import logger as torchrl_logger`
  (also `torchrl.torchrl_logger`).
- Timing: `torchrl.timeit`, never `time.time()` blocks.

## 4. TensorDict-first, TorchRL contracts everywhere

- New modules / transforms / losses / collectors / RB components accept and
  return `TensorDict` / `TensorDictBase`. No parallel dict-like containers.
- New objectives expose tensordict keys via `_AcceptedKeys` + `set_keys()`,
  matching existing losses.
- **Policies are `TensorDictModule`s** (`TensorDictModule`,
  `TensorDictModuleBase`, `ProbabilisticActor`, ...). Configuration goes into
  the constructor and `in_keys` / `out_keys` are declared on the module. A
  bare callable or a module-level function that computes an action is not a
  policy, in the library or in an example.
- **Rollouts are `env.rollout(policy=...)`** or a collector. No hand-written
  `reset` / `step` loops. Whatever is needed along the way (logging, extra
  observations, bookkeeping) is an env transform or an observation the env
  exposes; metrics are computed afterwards from the returned tensordict.
- **Aggregate with tensordict.** Stack per-episode or per-seed results into a
  `TensorDict` and reduce with `.mean(dim)`, `.min(dim)`,
  `.apply(fn, named=True)` and friends. Do not carry lists of dicts or
  dataclasses of floats and reduce them with comprehensions.
- **Env constructors are not kwarg soups.** Asset and simulator arguments
  (paths, backend, device, batch size) stay on the constructor. Task
  parameters (commands, reset distribution, reward weights, observation
  options) go into one config dataclass, with classmethods on the env that
  return presets (`MyEnv.some_task(...)`) so tasks have names. The built-in
  reward must be switchable off so a transform can supply another one.

## 5. Type hints & annotations

- Public signatures carry type hints. Hints must be accurate (not enforced by
  mypy, but wrong hints are worse than none), and must agree with the types
  stated in the docstring.
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

### 6a. Module device state

- **Do not define or assign `self.device` on an `nn.Module`**, or cache an
  equivalent single-device attribute. `module.to(...)`, `.cuda()` and
  `_apply(...)` move parameters and buffers, not Python state, so a cached
  device goes stale; and a module may span several devices (pipeline or
  tensor parallelism, FSDP, manual placement), in which case no single module
  device exists.
- Derive placement from the specific input, parameter or buffer involved in an
  operation (`tensor.new_*`, `device=tensor.device`). Do not infer a device
  for the whole module from its first parameter.
- Register persistent tensor state as a parameter or buffer so normal module
  transforms move it. A constructor may accept `device` to place initial
  state, but must not retain it as module state.
- Do not work around this by overriding `to()` or `_apply()` to synchronize a
  device cache; that still encodes an invalid single-device assumption.

## 7. Tests

- Every new public class / function needs tests.
- **Do not create new test files** when an existing one covers the area --
  extend it. Exception: a brand-new objective gets `test/test_<algo>.py`.
- If your module accepts a `NestedKey` input, add a test exercising a nested
  key (not just a flat string).
- Test files end with an `if __name__ == "__main__":` block that invokes
  `pytest.main(...)`, so the file can be executed directly.
- New algorithms: also tested in the sota-implementations CI.
- **Keep the volume proportionate.** One test per behavior, not one per
  attribute; several hundred lines of tests for one class is a signal to
  consolidate. Assert through the public API; tests that read or mutate
  private attributes lock in the implementation instead of the behavior.

### 7a. Behavioral regression tests

- **Test the bug fix, not its implementation.** A regression test must assert
  the behavior that was broken and fail if the bug is reintroduced.
- **Use mocks and monkeypatching sparingly.** A test that only proves an
  internal method was called is incomplete. For a retry bug, make the
  dependency fail once and then succeed, and assert that the operation
  returns the expected result; do not only mock `_retry()` and assert that it
  was called.
- Prefer small deterministic inputs and expected results derived independently
  from the code under test. Shape, finiteness, key-presence and no-exception
  checks are not enough unless they are the behavior being fixed.
- Parametrize only when cases exercise distinguishable behavior. Every
  parameter must reach the code under test and affect an assertion.

### 7b. The `gpu` marker (load-bearing!)

The unified Linux CI (`.github/workflows/test-linux.yml`) collects tests with
**two mutually-exclusive marker filters**: `tests-cpu*` jobs run with
`-m 'not gpu'`, `tests-gpu*` and `tests-stable-gpu*` jobs run with `-m gpu`.

Any test gated by `@pytest.mark.skipif(not torch.cuda.is_available(), ...)`,
`@pytest.mark.skipif(not torch.cuda.device_count(), ...)`, the project's
`_has_triton` / `_has_cuda` flags, or any other CUDA-only requirement
**must** also carry `@pytest.mark.gpu`. Otherwise the CPU runners skip it,
the GPU runners deselect it before collection, and the test never runs in CI
and silently rots -- the exact failure mode that let the triton RNN
recurrent-matmul bug ship.

```python
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_something_cuda_specific():
    ...
```

Apply the marker at whatever scope is appropriate (function, class, or module
via `pytestmark = pytest.mark.gpu`); both decorators are required, in any
order. Dedicated GPU-runner workflows that pin a test file or `-k` filter
(e.g. `unittests-torch_geometric`, `unittests-isaaclab`) do not use the
`-m gpu` filter, so the marker is optional but recommended there. **Do not
add `@pytest.mark.gpu`** to tests that meaningfully exercise both CPU and GPU
paths via parametrization (e.g. `device = "cpu" if torch.cuda.device_count()
== 0 else "cuda"`); those must keep running on the CPU side.

### 7c. PR-gated CI suites: `ci/olddeps` and `ci/optdeps` labels

Two expensive suites in `.github/workflows/test-linux.yml` do NOT run on pull
requests by default; they run on every push to main and on the nightly
orchestrator. A PR that needs their signal opts in via a label:

- **`ci/olddeps`** runs `tests-olddeps` (oldest supported stable torch, cu118,
  gym 0.13). Add it whenever your change uses a torch API added recently (a
  new `torch.*` function, kwarg or behavior flag). If in doubt whether the
  oldest supported torch has it, add the label.
- **`ci/optdeps`** runs the full `tests-optdeps` suite (~2h). Add it when your
  change touches optional-dependency integrations or their import paths. PRs
  otherwise get `tests-optdeps-smoke`, which only builds the environment and
  checks imports.

Adding a label does NOT retrigger CI: apply the label first, then push a
commit or re-run the workflow so the gate sees it.

## 8. Documentation

- Every new public class / function referenced in `docs/source/reference/*.rst`.
- Sphinx-style docstrings. Positional parameters go under `Args:`,
  keyword-only ones under `Keyword Args:`. **Every entry states its type** in
  parentheses (`(float or Sequence[float], optional)`), its default, and what
  the value does. For a sequence or a mapping, say how it is consumed: sampled
  at every reset or fixed, whether duplicates matter, whether a tensordict
  entry can override it. A reader must not have to open the code to learn
  whether a float or a tuple is expected.
- **`Examples:` show real usage** a reader would copy: constructing the object,
  then what makes it interesting -- batching and devices, switching backends
  or compiling, setting a task, rendering, reading the outputs. A single
  `env.rollout(10)` placeholder does not count. Mark lines that need assets or
  hardware with `# doctest: +SKIP`; everything else must run.
- Paper references: include the arXiv link + short citation in the class
  docstring.
- No emojis anywhere -- code, docstrings, comments, commits, PR bodies.

## 9. Tutorials and examples

New "headline" features (algorithm family, collector, env wrapper) ship a
tutorial under `tutorials/` (or extend an existing one). Sphinx-first:

- `# prose comments` for explanation, **not** `print(...)`.
- Include "What you will learn", "Conclusion", "Further reading" sections
  (names can be rephrased), mirroring existing tutos.

Scripts under `examples/` and `sota-implementations/` are read as reference
code and are held to rule 4 with extra force:

- A few TorchRL components wired together, not a framework. Every function
  and class must justify its existence; if a docstring cannot say in one
  sentence why a reader wants it, remove it. Configuration lives on the policy
  and env constructors, not in extra dataclasses.
- **Do not commit notebooks (`.ipynb`), checkpoints, videos or datasets.**
  Document the command that generates them (e.g. `rlrender ... --format ipynb`).

## 10. Benchmarks

Performance-relevant changes (anything on a hot path: collectors, RB, losses,
transforms, env stepping) add/extend a benchmark under `benchmarks/`. Pure
correctness fixes don't need one.

## 11. SOTA implementations

New algorithm needs: a runnable script under `sota-implementations/<algo>/`
with a Hydra config, plus entries in `sota-check/` and in the
`test-linux-sota` CI smoke list at
`.github/unittest/linux_sota/scripts/test_sota.py`.

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

```text
[Algorithm] [BE] [BugFix] [CI] [Deprecation] [Doc]
[Feature] [Minor] [Performance] [Quality] [Refactor]
[Test] [Versioning]
```

Pick the most specific. No squash requirement on commits -- just make each
commit read sensibly on its own.

## 14. Config / class parity

Some classes (Trainers, losses, RB components, transforms, ...) have a Hydra
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
