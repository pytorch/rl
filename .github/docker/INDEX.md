# TorchRL Docker Images - Index

Quick reference guide to all documentation.

## 🚀 Getting Started (Start Here!)

**New to this?** Start with these in order:

1. **[SUMMARY.md](SUMMARY.md)** - Executive summary (5 min read)
2. **[QUICKSTART.md](QUICKSTART.md)** - Get running in 30 minutes
3. **[COMPARISON.md](COMPARISON.md)** - See the performance gains

## 📚 Documentation

### For Implementation
- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide (30 min to first image)
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Detailed migration steps
- **[PR_TEMPLATE.md](PR_TEMPLATE.md)** - Template for your PR

### For Understanding
- **[README.md](README.md)** - Architecture and strategy overview
- **[COMPARISON.md](COMPARISON.md)** - Performance analysis and ROI
- **[COST_ANALYSIS.md](COST_ANALYSIS.md)** - ⭐ **Detailed cost breakdown ($0!)** ⭐
- **[SUMMARY.md](SUMMARY.md)** - Complete overview

## 🐳 Docker Files

| File | Purpose | Base Image |
|------|---------|------------|
| `Dockerfile.base` | System dependencies | nvidia/cuda |
| `Dockerfile.nightly` | PyTorch nightly + deps | base |
| `Dockerfile.stable` | PyTorch stable + deps | base |
| `Dockerfile.habitat` | Habitat-sim support | nightly |

## 🛠️ Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `build.sh` | Build images locally | `./build.sh nightly --cuda-version 12.4.0 --python-version 3.11` |
| `test_image.sh` | Verify image works | `./test_image.sh ghcr.io/pytorch/torchrl-ci:nightly-cuda12.4-py3.11-latest` |
| `run_all_docker.sh` | Run tests in CI | Called by GitHub Actions |

## 🔄 Workflows

| Workflow | Purpose | Trigger |
|----------|---------|---------|
| `build-docker-images.yml` | Build and push images | Weekly cron, manual, on changes |
| `test-linux.yml` | Use images for testing | On PR, push to main |

## 📊 Performance Summary

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Setup time | 15-28 min | 3-6 min | **80% faster** |
| Total job | 45-68 min | 33-46 min | **27-32% faster** |
| Monthly hours saved | - | ~425 hours | **$200/month** |

## 🎯 Quick Reference

### Image Tags
```
ghcr.io/pytorch/torchrl-ci:base-cuda12.4-py3.11
ghcr.io/pytorch/torchrl-ci:nightly-cuda12.4-py3.11-latest
ghcr.io/pytorch/torchrl-ci:nightly-cuda12.4-py3.11-20250115
ghcr.io/pytorch/torchrl-ci:stable-cuda12.4-py3.10
```

### Build Commands
```bash
# Build base
./build.sh base --cuda-version 12.4.0 --python-version 3.11

# Build nightly
./build.sh nightly --cuda-version 12.4.0 --python-version 3.11

# Build and push
./build.sh nightly --cuda-version 12.4.0 --python-version 3.11 --push
```

### Test Commands
```bash
# Test image
./test_image.sh ghcr.io/pytorch/torchrl-ci:nightly-cuda12.4-py3.11-latest

# Interactive test
docker run -it ghcr.io/pytorch/torchrl-ci:nightly-cuda12.4-py3.11-latest bash
```

## 🗺️ Implementation Roadmap

```
Day 1: Build Images
├─ Build base images
├─ Build nightly images
└─ Test with test_image.sh

Day 2: Test CI
├─ Update one job
├─ Create PR
└─ Verify tests pass

Week 1: Roll Out
├─ Update all jobs in test-linux.yml
├─ Update other workflows
└─ Monitor performance

Week 2: Automate
├─ Enable weekly builds
├─ Set up cleanup
└─ Document maintenance
```

## 📋 Checklist

Use this checklist when implementing:

```
Setup Phase
□ Read SUMMARY.md
□ Read QUICKSTART.md
□ Build base image
□ Build nightly image
□ Run test_image.sh

Testing Phase
□ Update one CI job
□ Create test PR
□ Verify tests pass
□ Compare timing

Rollout Phase
□ Update remaining jobs
□ Monitor for issues
□ Enable weekly builds
□ Update documentation

Maintenance Phase
□ Set up cleanup policy
□ Train team
□ Monitor metrics
```

## ❓ FAQ

**Q: How long does this take to implement?**
A: Initial setup is 2-4 hours. After that, it's automated.

**Q: What if the image build fails?**
A: Check MIGRATION_GUIDE.md troubleshooting section.

**Q: How much storage does this use?**
A: ~60-80 GB with retention policy. Free for public repos.

**Q: Can I test locally?**
A: Yes! Use `./build.sh` and `./test_image.sh`.

**Q: What if tests fail with Docker image?**
A: Compare environment, check MIGRATION_GUIDE.md, test interactively.

**Q: How often are images updated?**
A: Nightly images weekly, stable images monthly, base images as needed.

## 🔗 Quick Links

- [GitHub Container Registry](https://github.com/pytorch/rl/pkgs/container/torchrl-ci)
- [Build Workflows](https://github.com/pytorch/rl/actions/workflows/build-docker-images.yml)
- [TorchRL CI Workflows](../.github/workflows/)
- [Docker Documentation](https://docs.docker.com/)

## 📞 Support

1. **Documentation**: Read the guides above
2. **Testing**: Use `test_image.sh` to debug
3. **Interactive**: `docker run -it <image> bash`
4. **Issues**: Open a GitHub issue with logs

---

**Last Updated**: 2025-10-15
**Version**: 1.0

