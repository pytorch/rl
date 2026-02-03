# TorchRL Custom Docker Images

This directory contains Dockerfiles for TorchRL CI to speed up test execution by pre-installing dependencies.

## Strategy

We maintain multiple Docker images for different test configurations:

1. **Base images** (rarely change):
   - System dependencies (OpenGL, GLFW, etc.)
   - Python environment with uv
   - Common test dependencies

2. **PyTorch-versioned images** (updated weekly):
   - Base image + PyTorch (stable/nightly)
   - tensordict (stable/nightly)
   - Built TorchRL C++ extensions (when applicable)

3. **Specialized images** (as needed):
   - Images for specific environment tests (Habitat, LLM, etc.)

## Image Naming Convention

```
ghcr.io/pytorch/torchrl-ci:<tag>
```

Tags:
- `base-cuda12.4-py3.11` - Base image with system deps
- `nightly-cuda12.4-py3.11-20250115` - Nightly PyTorch + date
- `stable-cuda12.4-py3.11` - Stable PyTorch
- `habitat-cuda12.1-py3.9` - Specialized for Habitat tests

## Building Images

### Locally (for testing)
```bash
cd .github/docker
docker build -f Dockerfile.base -t torchrl-ci:base-cuda12.4-py3.11 \
  --build-arg CUDA_VERSION=12.4.0 \
  --build-arg PYTHON_VERSION=3.11 .
```

### Via GitHub Actions
Push to the `docker-images` branch or trigger the workflow manually.

## Updating Images

### Weekly (automated via cron):
- Rebuild nightly images with latest PyTorch
- Tag with date: `nightly-cuda12.4-py3.11-20250115`

### Monthly or as needed:
- Rebuild base images when system dependencies change
- Rebuild stable images when new PyTorch release

### On PR (optional):
- Build test images for major dependency changes

## CI Usage

In workflow files, replace:
```yaml
docker-image: "nvidia/cuda:12.2.0-devel-ubuntu22.04"
```

With:
```yaml
docker-image: "ghcr.io/pytorch/torchrl-ci:nightly-cuda12.4-py3.11"
```

## Storage & Cleanup

- Images are stored in GitHub Container Registry (ghcr.io)
- **Storage cost: $0** - Container Registry is free for all users ([source](https://docs.github.com/en/billing/concepts/product-billing/github-packages))
- Keep last 3 nightly images per configuration (~7 days)
- Keep last 2 stable images per configuration
- Automated cleanup via GitHub Actions (optional, not required for cost reasons)

## Expected Speedup

Before: 15-25 minutes overhead per job
After: 1-3 minutes overhead per job
**Savings: 12-22 minutes per job, ~80% reduction in setup time**

