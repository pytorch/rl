# Flaky Test Resolution Guide

This guide provides step-by-step instructions for identifying, debugging, and fixing flaky tests in the TorchRL codebase.

## Overview

A flaky test is one that fails intermittently without code changes. The TorchRL CI automatically tracks flaky tests using the [Flaky Test Dashboard](https://docs.pytorch.org/rl/flaky/). Tests are considered flaky if they have a failure rate between 5% and 95% with at least 2 failures across recent CI runs.

## Step 1: Gather Flaky Test Information

### From the GitHub Issue (Recommended Starting Point)

The CI automatically maintains a GitHub issue with the complete flaky test report:

**Find the current issue:**
```bash
gh issue list --repo pytorch/rl --label flaky-test-tracker --state open
```

Or browse directly: https://github.com/pytorch/rl/issues?q=label%3Aflaky-test-tracker+is%3Aopen

The issue contains:
- Full test nodeids (not truncated like in the dashboard)
- Failure rates and flaky scores
- List of newly flaky tests (last 7 days)

### From the Dashboard

Visit https://docs.pytorch.org/rl/flaky/ to see a visual overview with trend charts. The dashboard shows:

- **Test nodeid**: The pytest node identifier (may be truncated for long names)
- **Failure Rate**: Percentage of runs where the test failed
- **Flaky Score**: A confidence score (0-1) indicating how flaky the test is
- **Recent Failures**: Dates of the most recent failures

Note: The dashboard currently does not link directly to failing workflow runs. To find the actual error messages, you need to search the CI logs (see below).

### Get the Exact Error Message

To fix a flaky test, you need the actual error. The dashboard and issue show which tests are flaky, but not the error messages. Find them in the CI logs:

```bash
# List recent failed runs for the main test workflow
gh run list --repo pytorch/rl --workflow=test-linux.yml --status=failure --limit=10

# View the failed logs for a specific run
gh run view <run-id> --repo pytorch/rl --log-failed

# Search for a specific test in recent runs
gh run list --repo pytorch/rl --workflow=test-linux.yml --limit=20 --json databaseId,conclusion,createdAt \
  | jq '.[] | select(.conclusion == "failure")'
```

**Finding the right workflow:** Check the "Recent Failures" dates in the flaky report, then look for failed runs around those dates in the relevant workflow:
- `test-linux.yml` - Most common (CPU + GPU tests)
- `test-linux-libs.yml` - External library tests
- `test-linux-llm.yml` - LLM tests
- `test-linux-habitat.yml` - Habitat tests
- `test-linux-sota.yml` - SOTA implementation tests

Alternatively, browse the Actions tab: https://github.com/pytorch/rl/actions and filter by workflow and status.

## Step 2: Identify Non-Deterministic Failure Causes

Common causes of flaky tests in TorchRL:

### Random Seed Issues
RL algorithms are inherently stochastic. Tests may fail when:
- Random seeds are not set or not set early enough
- Seeds don't cover all sources of randomness (numpy, torch, gym environments)

### Race Conditions
Multiprocessing tests (collectors, distributed) can have:
- Timing-dependent behavior
- Shared state between workers
- Improper synchronization

### Resource Contention
- GPU memory not properly freed between tests
- File handles or ports not released
- Temporary files conflicting

### Timing-Dependent Assertions
- Assertions on execution time
- Polling/waiting with insufficient timeouts
- Async operations not properly awaited

### Test Ordering/Contamination
- Shared global state modified by earlier tests
- Module-level caching not reset
- Environment variables persisting

### External Dependencies
- Network requests (downloading models, gym environments)
- External services being unavailable
- Version mismatches in dependencies

## Step 3: Implement the Fix

### Important: Never Skip or xfail Flaky Tests

**Skipping tests (`@pytest.mark.skip`) or marking them as expected failures (`@pytest.mark.xfail`) is never an acceptable solution for flaky tests.**

A flaky test indicates a flaky feature. The test is doing its job by exposing non-deterministic behavior in the code. The correct approach is to fix the underlying issue in the feature itself, not to hide it.

If a test is flaky because of a race condition in a collector, fix the collector. If it's flaky because of improper state management, fix the state management. The test should pass reliably once the feature works reliably.

### Common Fixes

**Set seeds explicitly:**
```python
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
# For gym environments:
env.reset(seed=42)
```

**Add proper synchronization:**
```python
# Wait for all processes
torch.distributed.barrier()

# Use timeouts with retries
for _ in range(max_retries):
    try:
        result = operation_with_timeout()
        break
    except TimeoutError:
        continue
```

**Increase timeouts for slow operations:**
```python
@pytest.mark.timeout(120)  # Increase from default
def test_slow_operation():
    ...
```

**Isolate tests that modify global state:**
```python
@pytest.fixture(autouse=True)
def reset_global_state():
    original = some_module.GLOBAL_VAR
    yield
    some_module.GLOBAL_VAR = original
```

## Step 4: Open a PR with Test Repetition

### Fixing Multiple Flaky Tests with ghstack

When fixing multiple flaky tests, use [ghstack](https://github.com/ezyang/ghstack) to create one PR per fix. This keeps changes isolated and makes review easier:

```bash
# Create one commit per fix with descriptive names
git commit -m "[BugFix] Fix race condition in LazyMemmapStorage cleanup"
git commit -m "[BugFix] Add proper synchronization to MultiKeyEnv rollout"
git commit -m "[BugFix] Set random seed in SAC prioritized weights test"

# Push all commits as separate stacked PRs
ghstack
```

Each commit becomes its own PR, allowing:
- Independent review and CI runs for each fix
- Easy bisection if one fix causes regressions
- Parallel merging once each fix is verified

### Validating Your Fix

To validate your fix, you need to run the test multiple times. Since flaky tests fail intermittently, a single successful run doesn't prove the fix works.

### Modify the Test Command (Recommended)

Edit `.github/unittest/linux/scripts/run_all.sh` to run only the flaky test multiple times:

```bash
# Add this before or instead of the normal test run
for i in {1..20}; do
  echo "=== Run $i of 20 ==="
  pytest test/test_file.py::TestClass::test_method -v || exit 1
done
```

Or use `pytest-repeat` (already installed):

```bash
pytest test/test_file.py::TestClass::test_method --count=20 -v
```

### Example PR Modification

In your PR, temporarily modify the test script:

```bash
# In run_all.sh, replace the normal pytest call with:
pytest test/test_envs.py::TestMyFlaky -v --count=20 --timeout=60
```

### Note on Separate CI Workflows

Running flaky tests in a dedicated workflow can be faster, but has complications:
- Need to match the original platform (Linux, macOS, Windows)
- Need to match GPU/CPU configuration
- Some flaky tests fail due to test contamination (other tests running first), which a separate workflow won't catch

For these reasons, running repeated tests in the existing workflow is usually more reliable.

## Step 5: Monitor the CI Run

Use `gh` to monitor your PR's CI status:

```bash
# List runs for your PR
gh run list --branch=<your-branch> --limit=5

# Watch a run in real-time
gh run watch <run-id>

# View details of a completed run
gh run view <run-id>

# View only failed job logs
gh run view <run-id> --log-failed

# Download artifacts (including test results JSON)
gh run download <run-id> -n test-results-cpu-3.12
```

### Check Specific Jobs

```bash
# List jobs in a run
gh run view <run-id> --json jobs --jq '.jobs[] | "\(.name): \(.status) \(.conclusion)"'
```

## Step 6: Verify the Fix

### How Many Runs Are Needed?

The number of successful runs needed depends on the original failure rate:

| Original Failure Rate | Runs Needed for 95% Confidence | Runs Needed for 99% Confidence |
|----------------------|-------------------------------|-------------------------------|
| 50% | 5 | 7 |
| 20% | 14 | 21 |
| 10% | 29 | 44 |
| 5% | 59 | 90 |

**Formula:** To achieve confidence level C that the test is fixed:
- Number of runs N = log(1 - C) / log(1 - p)
- Where p = original failure rate

**Practical recommendation:** Run the test 10-20 times for tests with >10% failure rate, or 30-50 times for tests with lower failure rates.

### What If It Still Fails?

If the test still fails during repeated runs:
1. Check if it's a different failure mode
2. Look for additional sources of non-determinism
3. Consider if the test needs fundamental redesign

## Step 7: Cleanup Before Merging

Before your PR is ready to merge:

1. **Remove test repetition**: Revert changes to `.github/unittest/linux/scripts/run_all.sh`
2. **Remove temporary markers**: Remove any `@pytest.mark.flaky` decorators added during debugging
3. **Update PR description**: Document what caused the flakiness and how you fixed it
4. **Verify normal CI passes**: Ensure the full test suite still passes

## Reference

### Key Resources

| Resource | Purpose |
|----------|---------|
| [Flaky Test Issue](https://github.com/pytorch/rl/issues?q=label%3Aflaky-test-tracker+is%3Aopen) | GitHub issue with full flaky test report (updated daily) |
| [Flaky Test Dashboard](https://docs.pytorch.org/rl/flaky/) | Visual dashboard with trend charts |
| [GitHub Actions](https://github.com/pytorch/rl/actions) | CI runs with error logs |

### Key Files

| File | Purpose |
|------|---------|
| `.github/workflows/flaky-test-tracker.yml` | Workflow that analyzes and reports flaky tests |
| `.github/scripts/analyze_flaky_tests.py` | Script that identifies flaky tests from CI history |
| `.github/unittest/linux/scripts/run_all.sh` | Main test runner script |

### Workflows Analyzed for Flaky Tests

The flaky test tracker monitors these workflows:
- `test-linux.yml` - Main Linux tests (CPU and GPU)
- `test-linux-libs.yml` - Tests with external libraries
- `test-linux-habitat.yml` - Habitat environment tests
- `test-linux-llm.yml` - LLM-related tests
- `test-linux-sota.yml` - SOTA implementation tests

### Flaky Test Detection Thresholds

From `.github/scripts/analyze_flaky_tests.py`:
- **Minimum failure rate**: 5% (below this = probably fixed)
- **Maximum failure rate**: 95% (above this = broken, not flaky)
- **Minimum failures**: 2 (need at least 2 failures to flag)
- **Minimum executions**: 3 (need enough data points)
- **New flaky window**: 7 days (tests that became flaky recently)
