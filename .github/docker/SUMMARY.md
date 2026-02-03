# Docker Images for TorchRL CI - Summary

## Executive Summary

**Problem**: TorchRL CI jobs spend 15-28 minutes (60-80% of total time) installing dependencies from scratch on every run.

**Solution**: Pre-build Docker images with all dependencies, reducing setup time to 3-6 minutes.

**Impact**: 
- 27-32% faster CI runs
- ~425 runner hours saved per month
- ~$200/month cost savings
- Better reliability and developer experience

**Investment**: 2-4 hours initial setup, 5 min/week maintenance (automated)

**ROI**: Positive within first week

## What Was Created

### Documentation (6 files)
1. **README.md** - Overview of strategy and naming conventions
2. **QUICKSTART.md** - Get started in 30 minutes
3. **MIGRATION_GUIDE.md** - Detailed migration steps and troubleshooting
4. **COMPARISON.md** - Performance analysis and cost comparison
5. **PR_TEMPLATE.md** - Template for implementation PR
6. **SUMMARY.md** - This file

### Docker Infrastructure (5 files)
1. **Dockerfile.base** - System dependencies and Python environment
2. **Dockerfile.nightly** - PyTorch nightly + test dependencies
3. **Dockerfile.stable** - PyTorch stable + test dependencies  
4. **Dockerfile.habitat** - Specialized for Habitat tests
5. **10_nvidia.json** - NVIDIA EGL configuration

### Build & Test Scripts (3 files)
1. **build.sh** - Build images locally
2. **test_image.sh** - Verify image functionality
3. **.dockerignore** - Optimize Docker context

### CI Integration (2 files)
1. **build-docker-images.yml** - Automated image building workflow
2. **run_all_docker.sh** - Simplified test script for Docker images

## Image Architecture

```
nvidia/cuda:12.4.0-devel-ubuntu22.04  (NVIDIA base)
         ↓
    Dockerfile.base
    - System packages (OpenGL, build tools)
    - uv package manager
    - Environment variables
         ↓
    ┌────────────────┬────────────────┐
    ↓                ↓                ↓
Dockerfile.nightly  Dockerfile.stable  Dockerfile.habitat
- Python env       - Python env       - Python env
- PyTorch nightly  - PyTorch stable   - PyTorch nightly
- tensordict git   - tensordict PyPI  - tensordict git
- Test deps        - Test deps        - Habitat-sim
- RL libraries     - RL libraries     - Habitat-lab
```

## Image Naming Convention

```
ghcr.io/pytorch/torchrl-ci:<tag>

Tags:
- base-cuda12.4-py3.11
- nightly-cuda12.4-py3.11-20250115
- nightly-cuda12.4-py3.11-latest
- stable-cuda12.4-py3.10
- habitat-cuda12.1-py3.9
```

## Usage in CI Workflows

### Before
```yaml
docker-image: "nvidia/cuda:12.2.0-devel-ubuntu22.04"
script: |
  bash .github/unittest/linux/scripts/run_all.sh
```

### After
```yaml
docker-image: "ghcr.io/pytorch/torchrl-ci:nightly-cuda12.4-py3.11-latest"
script: |
  bash .github/unittest/linux/scripts/run_all_docker.sh
```

## Implementation Checklist

### Phase 1: Build Images (Day 1)
- [ ] Review all Docker files
- [ ] Build base images for required CUDA versions
- [ ] Build nightly images on top of base
- [ ] Push images to ghcr.io
- [ ] Test images with test_image.sh

### Phase 2: Test with One Job (Day 2)
- [ ] Update one CI job to use Docker image
- [ ] Create PR and run CI
- [ ] Compare timing with baseline
- [ ] Verify all tests pass

### Phase 3: Roll Out (Week 1)
- [ ] Update remaining jobs in test-linux.yml
- [ ] Update test-linux-habitat.yml
- [ ] Update test-linux-libs.yml
- [ ] Update other workflows as needed
- [ ] Monitor for issues

### Phase 4: Automate (Week 2)
- [ ] Enable weekly cron job
- [ ] Set up image cleanup policy
- [ ] Document maintenance procedures
- [ ] Train team on updates

## Maintenance Schedule

### Weekly (Automated)
- **Sunday 2 AM UTC**: Rebuild nightly images with latest PyTorch
- Auto-tag with date: `nightly-cuda12.4-py3.11-20250115`
- Auto-cleanup images older than 7 days

### Monthly (Manual)
- Review and update base images if needed
- Rebuild stable images after PyTorch releases
- Check storage usage and cleanup if needed

### As Needed
- Update images when dependencies change
- Build specialized images for new test types
- Adjust retention policy based on usage

## Key Commands

### Build Images Locally
```bash
cd .github/docker
./build.sh base --cuda-version 12.4.0 --python-version 3.11
./build.sh nightly --cuda-version 12.4.0 --python-version 3.11
```

### Test Images
```bash
./test_image.sh ghcr.io/pytorch/torchrl-ci:nightly-cuda12.4-py3.11-latest
```

### Push to Registry
```bash
./build.sh nightly --cuda-version 12.4.0 --python-version 3.11 --push
```

### Use in CI
```yaml
docker-image: "ghcr.io/pytorch/torchrl-ci:nightly-cuda12.4-py3.11-latest"
```

## Troubleshooting

### Image build fails
1. Check Docker logs: `docker logs <container>`
2. Build with --no-cache: `./build.sh nightly --no-cache`
3. Test dependencies manually in container

### Tests fail with Docker image
1. Compare environment: `docker run <image> env`
2. Check paths and permissions
3. Verify all dependencies present: `./test_image.sh <image>`

### Image too large
1. Clean up in same RUN layer
2. Remove unnecessary files
3. Use multi-stage builds if needed

### Can't pull image
1. Check permissions in ghcr.io
2. Verify image tag exists
3. Try manual docker pull

## Performance Metrics to Track

1. **CI timing** (most important)
   - Before: 45-68 min per job
   - Target: 33-46 min per job
   - Measure: Wall clock time from GitHub Actions

2. **Setup timing**
   - Before: 15-28 min
   - Target: 3-6 min
   - Measure: Time before first pytest runs

3. **Image sizes**
   - Base: ~3 GB
   - Nightly: ~8 GB
   - Monitor: Don't exceed 10 GB per image

4. **Build success rate**
   - Target: >95% success rate
   - Monitor: Weekly build workflow

5. **Storage usage**
   - Target: <100 GB total
   - Monitor: ghcr.io dashboard

## Success Criteria

After implementation, you should see:
- ✅ CI jobs complete 12-22 minutes faster
- ✅ Setup takes 3-6 minutes instead of 15-28 minutes
- ✅ All tests pass as before
- ✅ Weekly builds succeed automatically
- ✅ No significant increase in test failures
- ✅ Positive feedback from developers

## Next Steps

1. **Read QUICKSTART.md** - Get started in 30 minutes
2. **Build first image** - Test locally or via GitHub Actions
3. **Test with one job** - Create test PR
4. **Review COMPARISON.md** - See detailed performance analysis
5. **Follow MIGRATION_GUIDE.md** - Roll out to all jobs
6. **Use PR_TEMPLATE.md** - Create implementation PR

## Resources

- Docker documentation: https://docs.docker.com/
- GitHub Container Registry: https://docs.github.com/packages/working-with-a-github-packages-registry/working-with-the-container-registry
- GitHub Actions: https://docs.github.com/actions
- TorchRL CI: `.github/workflows/`

## Questions?

For questions or issues:
1. Check MIGRATION_GUIDE.md troubleshooting section
2. Review COMPARISON.md for performance analysis
3. Test images locally with test_image.sh
4. Open an issue with logs and details

---

**Created**: 2025-10-15
**Last Updated**: 2025-10-15
**Status**: Ready for implementation

