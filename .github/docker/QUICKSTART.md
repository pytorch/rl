# Quick Start Guide

This guide will help you get started with custom Docker images for TorchRL CI in under 30 minutes.

## Prerequisites

- Docker installed locally (for testing)
- GitHub repository write access
- Access to push to GitHub Container Registry

## 1. Initial Setup (5 minutes)

### Enable GitHub Container Registry

1. Go to repository Settings → Packages
2. Ensure packages can be published
3. Set package visibility to Public (for easier access)

### Configure Permissions

The workflow already uses `GITHUB_TOKEN` with `packages: write` permission, so no additional setup needed.

## 2. Build First Image (15 minutes)

### Option A: Via GitHub Actions (Recommended)

1. Push the Docker files to your repository:
   ```bash
   git add .github/docker/
   git add .github/workflows/build-docker-images.yml
   git commit -m "Add Docker image infrastructure"
   git push origin main
   ```

2. Manually trigger the workflow:
   - Go to Actions → "Build Docker Images"
   - Click "Run workflow"
   - Select "base" as image type
   - Click "Run workflow"

3. Wait ~10 minutes for base image to build

4. Trigger again for "nightly" images

### Option B: Build Locally (Faster for Testing)

```bash
cd .github/docker

# Build base image
docker build -f Dockerfile.base -t test-base:local \
  --build-arg CUDA_VERSION=12.4.0 \
  --build-arg PYTHON_VERSION=3.11 .

# Test it
docker run -it test-base:local bash
# Inside container, verify:
# - uv --version
# - ls /usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

## 3. Test Image Locally (5 minutes)

```bash
# Build nightly image on top of base
docker build -f Dockerfile.nightly -t test-nightly:local \
  --build-arg CUDA_VERSION=12.4.0 \
  --build-arg PYTHON_VERSION=3.11 \
  --build-arg CU_VERSION=cpu \
  --build-arg BASE_TAG=test-base:local .

# Test dependencies
docker run -it test-nightly:local bash
# Inside container:
python -c "import torch; print(torch.__version__)"
python -c "import tensordict; print(tensordict.__version__)"
python -c "import gymnasium"
```

## 4. Update One Workflow (5 minutes)

Create a test branch and update a single job:

```bash
git checkout -b docker-ci-test
```

Edit `.github/workflows/test-linux.yml`:

```yaml
# Find the tests-cpu job and update it:
jobs:
  tests-cpu:
    strategy:
      matrix:
        python_version: ["3.11"]  # Start with one version
      fail-fast: false
    uses: pytorch/test-infra/.github/workflows/linux_job_v2.yml@main
    with:
      runner: linux.12xlarge
      repository: pytorch/rl
      # OLD: docker-image: "nvidia/cuda:12.2.0-devel-ubuntu22.04"
      # NEW:
      docker-image: "ghcr.io/pytorch/torchrl-ci:nightly-cuda12.4-py3.11-latest"
      timeout: 90
      script: |
        if [[ "${{ github.ref }}" =~ release/* ]]; then
          export RELEASE=1
          export TORCH_VERSION=stable
        else
          export RELEASE=0
          export TORCH_VERSION=nightly
        fi
        export TD_GET_DEFAULTS_TO_NONE=1
        export PYTHON_VERSION=${{ matrix.python_version }}
        export CU_VERSION="cpu"
        
        # OLD: bash .github/unittest/linux/scripts/run_all.sh
        # NEW:
        bash .github/unittest/linux/scripts/run_all_docker.sh
```

## 5. Test & Measure (5 minutes)

```bash
git add .github/workflows/test-linux.yml
git commit -m "test: Use Docker image for tests-cpu job"
git push origin docker-ci-test
```

Create a PR and watch the CI run. Compare timing:

**Before**: Setup ~15-25 min + Tests ~30 min = **45-55 min total**
**After**: Setup ~1-3 min + Tests ~30 min = **31-33 min total**

## What's Next?

### If Tests Pass ✅

1. **Roll out to more jobs**: Update other jobs in the same workflow
2. **Enable automation**: Let the weekly cron job handle rebuilds
3. **Monitor**: Check image sizes and build times

### If Tests Fail ❌

Common issues and fixes:

1. **Import errors**: Missing dependency in Dockerfile
   - Add to `RUN uv pip install ...` line
   - Rebuild image

2. **Permission errors**: Path issues in container
   - Check `WORKDIR` in Dockerfile
   - Verify paths in test script

3. **GPU errors**: CUDA version mismatch
   - Verify CUDA version in Dockerfile matches runner
   - Check `CU_VERSION` environment variable

## Full Rollout Checklist

- [ ] Base images built for all CUDA versions (12.4, 11.8)
- [ ] Base images built for all Python versions (3.9, 3.10, 3.11, 3.12)
- [ ] Nightly images built and tested
- [ ] Stable images built (optional for now)
- [ ] One workflow job updated and tested
- [ ] All workflow jobs updated
- [ ] Weekly automation enabled
- [ ] Image cleanup policy implemented
- [ ] Documentation updated

## Maintenance Commands

```bash
# View built images
docker images | grep torchrl-ci

# Clean up local images
docker rmi $(docker images -q 'ghcr.io/pytorch/torchrl-ci')

# Rebuild specific image
cd .github/docker
./build.sh nightly --cuda-version 12.4.0 --python-version 3.11

# Test image interactively
docker run -it ghcr.io/pytorch/torchrl-ci:nightly-cuda12.4-py3.11-latest bash
```

## Cost Estimate

### Storage (GitHub Container Registry):
- Per image: 3-8 GB
- Total for 6 images: ~30-40 GB
- **Cost: $0** (Container Registry is free + TorchRL is public)
  - [Source](https://docs.github.com/en/billing/concepts/product-billing/github-packages): "Container image storage and bandwidth for the Container registry is currently free"

### Compute Savings:
- Per CI run: Save 12-22 minutes
- Per day (assuming 10 runs): Save 2-3.5 hours
- Per month: Save 60-100 runner hours
- Cost savings: **Significant** (depends on runner cost)

### Build Time Investment:
- Initial setup: ~2 hours
- Weekly maintenance: ~5 minutes (automated)
- ROI: **Positive within first week**

## Support

For issues or questions:
1. Check MIGRATION_GUIDE.md for troubleshooting
2. Review Docker logs: `docker logs <container>`
3. Test locally before pushing changes
4. Open an issue with logs and Dockerfile changes

