# CI Performance Comparison

## Current CI Setup (Without Docker Images)

### Time Breakdown - Single Job

| Step | Time | Description |
|------|------|-------------|
| apt-get update & upgrade | 2-3 min | System package updates |
| Install system libraries | 1-2 min | OpenGL, GLFW, build tools |
| Install uv | 0.5-1 min | Package manager |
| Create Python environment | 0.5-1 min | Virtual environment setup |
| Install test dependencies | 5-8 min | pytest, gymnasium, dm_control, etc. |
| Install PyTorch | 2-5 min | Download & install (varies by version) |
| Install tensordict | 1-3 min | From git or PyPI |
| Build TorchRL C++ extensions | 2-3 min | Compilation |
| Install TorchRL Python | 0.5-1 min | Package installation |
| **Setup Total** | **15-28 min** | |
| Run tests | 30-40 min | Actual test execution |
| **Job Total** | **45-68 min** | |

### Full CI Run

For a typical PR with 5 parallel jobs:
- **Total setup time**: 75-140 minutes (15-28 min × 5 jobs)
- **Total test time**: 150-200 minutes (30-40 min × 5 jobs)
- **Wall clock time**: 45-68 minutes (longest job, parallel execution)

## With Docker Images

### Time Breakdown - Single Job

| Step | Time | Description |
|------|------|-------------|
| Pull Docker image | 0.5-2 min | Cached after first pull |
| Build TorchRL C++ extensions | 2-3 min | Only non-cached step |
| Install TorchRL Python | 0.5-1 min | Package installation |
| **Setup Total** | **3-6 min** | |
| Run tests | 30-40 min | Actual test execution |
| **Job Total** | **33-46 min** | |

### Full CI Run

For a typical PR with 5 parallel jobs:
- **Total setup time**: 15-30 minutes (3-6 min × 5 jobs)
- **Total test time**: 150-200 minutes (30-40 min × 5 jobs)
- **Wall clock time**: 33-46 minutes (longest job, parallel execution)

## Savings Analysis

| Metric | Before | After | Savings |
|--------|--------|-------|---------|
| Setup time per job | 15-28 min | 3-6 min | **12-22 min** |
| Total job time | 45-68 min | 33-46 min | **12-22 min** |
| Setup time (5 jobs) | 75-140 min | 15-30 min | **60-110 min** |
| Wall clock time | 45-68 min | 33-46 min | **12-22 min (27-32%)** |

### Monthly Savings (Estimated)

Assuming:
- 10 PRs per day
- 5 jobs per PR
- Average 17 minutes saved per job

**Daily savings**: 10 PRs × 5 jobs × 17 min = **850 minutes/day** (~14 hours)
**Monthly savings**: 850 × 30 = **25,500 minutes/month** (~425 hours)

### Cost Savings

GitHub-hosted runner costs (estimated):
- Linux runner: ~$0.008/minute
- Monthly runner cost before: ~$0.008 × (10 PRs × 5 jobs × 60 min × 30 days) = **~$720/month**
- Monthly runner cost after: ~$0.008 × (10 PRs × 5 jobs × 43 min × 30 days) = **~$516/month**
- **Monthly savings: ~$204** (28% reduction)

Note: Actual costs vary based on runner type and GitHub plan.

## Side-by-Side Comparison

### Job: tests-cpu (Python 3.11)

#### Current Setup
```yaml
docker-image: "nvidia/cuda:12.2.0-devel-ubuntu22.04"
script: |
  # Install everything from scratch
  apt-get update && upgrade              # 2-3 min
  apt-get install system packages        # 1-2 min
  curl install uv                        # 1 min
  uv venv and install deps               # 6-9 min
  uv pip install torch                   # 2-5 min
  uv pip install tensordict              # 1-3 min
  uv pip install -e . --no-deps torchrl  # 2-4 min
  pytest test/                           # 30-40 min
  ----------------------------------------
  Total: 45-68 min
```

#### With Docker Image
```yaml
docker-image: "ghcr.io/pytorch/torchrl-ci:nightly-cuda12.4-py3.11-latest"
script: |
  # Most dependencies pre-installed
  source /opt/venv/bin/activate          # instant
  uv pip install -e . --no-deps torchrl  # 2-4 min
  pytest test/                           # 30-40 min
  ----------------------------------------
  Total: 33-46 min
  Savings: 12-22 min (27-32%)
```

## Real-World Impact

### Developer Experience

**Before**:
- Push commit → Wait 50+ min → See results
- Fix found → Push fix → Wait 50+ min again
- **Iteration cycle**: Very slow

**After**:
- Push commit → Wait 35 min → See results
- Fix found → Push fix → Wait 35 min again
- **Iteration cycle**: 30% faster

### CI Queue Time

**Before**:
- Long-running jobs occupy runners longer
- More queuing during peak times
- Higher chance of timeout (90 min limit)

**After**:
- Shorter jobs free up runners faster
- Less queuing overall
- More buffer before timeout

### Reliability

**Before**:
- More network calls = more failure points
- PyPI/GitHub/apt.com downtime affects CI
- Package version conflicts more common

**After**:
- Fewer external dependencies per run
- More resilient to external service issues
- Consistent environment across runs

## Image Storage Impact

### Storage Requirements

| Image Type | Size | Frequency | Count | Total |
|------------|------|-----------|-------|-------|
| Base | ~3 GB | Monthly | 4 | 12 GB |
| Nightly | ~8 GB | Weekly | 12 | 96 GB |
| Stable | ~8 GB | Monthly | 4 | 32 GB |
| **Total** | | | | **~140 GB** |

With retention policy (keep last 3 nightly per config):
- **~60-80 GB** maintained long-term

### GitHub Container Registry Pricing

According to [GitHub's official documentation](https://docs.github.com/en/billing/concepts/product-billing/github-packages):

#### ✅ Container Registry: **COMPLETELY FREE**
- **Storage**: Unlimited (free)
- **Bandwidth**: Unlimited (free)
- **Policy**: "Container image storage and bandwidth for the Container registry is currently free"
- **Notice**: GitHub will provide at least 1 month advance notice if this changes

#### ✅ Public Repositories: **ALWAYS FREE**
- TorchRL is a public repo (`pytorch/rl`)
- Even if Container Registry policy changes, public packages remain free
- No storage or transfer limits for public packages

**For TorchRL: $0/month storage cost** ✅

## Conclusion

**Total Benefits**:
- ✅ 27-32% faster CI runs
- ✅ ~425 runner hours saved per month
- ✅ More reliable (fewer external dependencies)
- ✅ Better developer experience
- ✅ Lower runner costs (~$200/month saved)
- ✅ Free storage (public repo)

**Costs**:
- Initial setup: ~2-4 hours
- Weekly maintenance: ~5 minutes (automated)
- Storage: $0 for public repos

**ROI**: **Positive within first week**

## Recommendation

**Strongly recommended** to implement Docker images for TorchRL CI. The benefits far outweigh the minimal setup and maintenance costs.

### Phased Rollout

1. **Week 1**: Build base and nightly images, test with one job
2. **Week 2**: Roll out to all jobs if successful
3. **Week 3**: Enable automated weekly builds
4. **Week 4**: Fine-tune and document

**Total time investment**: 4-6 hours
**Ongoing maintenance**: ~5 min/week (automated)

