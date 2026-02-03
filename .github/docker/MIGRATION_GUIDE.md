# Migration Guide: Using Custom Docker Images

This guide explains how to migrate your CI workflows to use custom Docker images.

## Step-by-Step Migration

### 1. Build Initial Images

First, build and push the base images:

```bash
cd .github/docker

# Build base images for different CUDA versions
./build.sh base --cuda-version 12.4.0 --python-version 3.11 --push
./build.sh base --cuda-version 11.8.0 --python-version 3.9 --push

# Build nightly images on top of base
./build.sh nightly --cuda-version 12.4.0 --python-version 3.11 --push
./build.sh nightly --cuda-version 11.8.0 --python-version 3.9 --push

# Build stable images
./build.sh stable --cuda-version 12.4.0 --python-version 3.10 --push
```

**Note**: You'll need appropriate permissions to push to `ghcr.io/pytorch/torchrl-ci`.

### 2. Enable GitHub Container Registry

1. Go to your repository settings
2. Navigate to Packages
3. Enable GitHub Container Registry if not already enabled
4. Make packages public (or configure access as needed)

### 3. Update Workflow Files

#### Before (test-linux.yml):
```yaml
jobs:
  tests-cpu:
    uses: pytorch/test-infra/.github/workflows/linux_job_v2.yml@main
    with:
      docker-image: "nvidia/cuda:12.2.0-devel-ubuntu22.04"
      script: |
        export PYTHON_VERSION=3.11
        bash .github/unittest/linux/scripts/run_all.sh
```

#### After:
```yaml
jobs:
  tests-cpu:
    uses: pytorch/test-infra/.github/workflows/linux_job_v2.yml@main
    with:
      docker-image: "ghcr.io/pytorch/torchrl-ci:nightly-cuda12.4-py3.11-latest"
      script: |
        export PYTHON_VERSION=3.11
        bash .github/unittest/linux/scripts/run_all_docker.sh
```

### 4. Create Simplified Test Scripts

Create new test scripts that skip dependency installation:

**`.github/unittest/linux/scripts/run_all_docker.sh`**:
```bash
#!/usr/bin/env bash
set -euxo pipefail

# Activate pre-installed environment
source /opt/venv/bin/activate

# Navigate to repo
cd /workspace/rl

# Git setup
git config --global --add safe.directory '*'

# Start Xvfb for rendering
Xvfb :99 -screen 0 1024x768x24 &

# Environment setup
export PYTORCH_TEST_WITH_SLOW='1'
export MKL_THREADING_LAYER=GNU
export CKPT_BACKEND=torch
export MAX_IDLE_COUNT=100
export BATCHED_PIPE_TIMEOUT=60
export TD_GET_DEFAULTS_TO_NONE=1

# Install TorchRL (only thing that needs building each time)
echo "Installing TorchRL from source..."
uv pip install -e . --no-build-isolation --no-deps

# Optional: Install VC1 for GPU tests
if [ "${CU_VERSION:-}" != cpu ] ; then
  python -c "from torchrl.envs.transforms.vc1 import VC1Transform; VC1Transform.install_vc_models(auto_exit=True)"
fi

# Run tests
pytest test/smoke_test.py -v --durations 200
pytest test/smoke_test_deps.py -v --durations 200 -k 'test_gym or test_dm_control_pixels or test_dm_control or test_tb'

if [ "${CU_VERSION:-}" != cpu ] ; then
  python .github/unittest/helpers/coverage_run_parallel.py -m pytest test \
    --instafail --durations 200 -vv --capture no --ignore test/test_rlhf.py \
    --ignore test/llm \
    --timeout=120 --mp_fork_if_no_cuda
else
  python .github/unittest/helpers/coverage_run_parallel.py -m pytest test \
    --instafail --durations 200 -vv --capture no --ignore test/test_rlhf.py \
    --ignore test/test_distributed.py \
    --ignore test/llm \
    --timeout=120 --mp_fork_if_no_cuda
fi

coverage combine
coverage xml -i
```

### 5. Rollout Plan

#### Phase 1: Test One Workflow
- Start with `tests-cpu` job in `test-linux.yml`
- Build image and update workflow
- Verify tests pass and timing improves
- Monitor for any issues

#### Phase 2: Expand to More Workflows
- Update `tests-gpu` job
- Update `tests-optdeps` job
- Update other Linux workflows

#### Phase 3: Enable Automation
- Set up weekly automated builds via cron
- Implement image cleanup policy
- Document maintenance procedures

### 6. Maintenance

#### Weekly Tasks (Automated):
- Rebuild nightly images with latest PyTorch
- Tag with date for tracking
- Clean up images older than 7 days

#### Monthly Tasks:
- Review and update base images if system deps changed
- Rebuild stable images when new PyTorch releases

#### As Needed:
- Update images when dependencies change significantly
- Build specialized images for new test types

## Troubleshooting

### Problem: Image too large
**Solution**: Use multi-stage builds, clean up in same layer:
```dockerfile
RUN apt-get update && apt-get install -y package \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
```

### Problem: Cache not working
**Solution**: Ensure build order doesn't change, use explicit cache targets

### Problem: Nightly build broke
**Solution**: Use dated tags so you can roll back:
```yaml
docker-image: "ghcr.io/pytorch/torchrl-ci:nightly-cuda12.4-py3.11-20250110"
```

### Problem: Need to test dependency change
**Solution**: Build test image locally:
```bash
# Modify Dockerfile
./build.sh nightly --cuda-version 12.4.0 --python-version 3.11

# Test locally
docker run -it ghcr.io/pytorch/torchrl-ci:nightly-cuda12.4-py3.11-20250115 bash
```

## Expected Results

### Before Docker Images:
- **Setup time**: 15-25 minutes per job
- **Total CI time**: 90-120 minutes (with queue)
- **Cost**: High (more runner time)

### After Docker Images:
- **Setup time**: 1-3 minutes per job (just install TorchRL)
- **Total CI time**: 50-70 minutes (with queue)
- **Cost**: Lower (less runner time)
- **Savings**: ~40% reduction in CI time

## Cost Analysis

### Storage Costs:
- Base image: ~3 GB per configuration
- Nightly image: ~8 GB per configuration
- Total for 4 configurations: ~44 GB
- GitHub provides generous free storage for public repos

### Compute Savings:
- Save 12-20 minutes per job
- With 5-10 jobs per push: **60-200 minutes saved per push**
- Monthly savings: **Significant** (depends on push frequency)

**ROI**: Positive within first month for active repositories

