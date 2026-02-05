# Flaky Test Report - 2026-02-05

## Summary

- **Flaky tests**: 13
- **Newly flaky** (last 7 days): 13
- **Resolved**: 0
- **Total tests analyzed**: 26783
- **CI runs analyzed**: 161

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...b.py::TestLazyMemmapStorageCleanup::test_cleanup_registry` 🆕 | 5.3% (10/190) | 10 | 0.11 | 2026-01-29 |
| `...::TestRayCollector::test_ray_replaybuffer[None-None-None]` 🆕 | 5.3% (4/76) | 4 | 0.08 | 2026-01-30 |
| `...stRayCollector::test_ray_replaybuffer[None-None-storage1]` 🆕 | 5.3% (4/76) | 4 | 0.08 | 2026-01-30 |
| `...stRayCollector::test_ray_replaybuffer[None-sampler1-None]` 🆕 | 5.3% (4/76) | 4 | 0.08 | 2026-01-30 |
| `...yCollector::test_ray_replaybuffer[None-sampler1-storage1]` 🆕 | 5.3% (4/76) | 4 | 0.08 | 2026-01-30 |
| `...est_ray_replaybuffer[None-SamplerWithoutReplacement-None]` 🆕 | 5.3% (4/76) | 4 | 0.08 | 2026-01-30 |
| `...ray_replaybuffer[None-SamplerWithoutReplacement-storage1]` 🆕 | 5.3% (4/76) | 4 | 0.08 | 2026-01-30 |
| `...estRayCollector::test_ray_replaybuffer[writer1-None-None]` 🆕 | 5.3% (4/76) | 4 | 0.08 | 2026-01-30 |
| `...ayCollector::test_ray_replaybuffer[writer1-None-storage1]` 🆕 | 5.3% (4/76) | 4 | 0.08 | 2026-01-30 |
| `...ayCollector::test_ray_replaybuffer[writer1-sampler1-None]` 🆕 | 5.3% (4/76) | 4 | 0.08 | 2026-01-30 |
| `...llector::test_ray_replaybuffer[writer1-sampler1-storage1]` 🆕 | 5.3% (4/76) | 4 | 0.08 | 2026-01-30 |
| `..._ray_replaybuffer[writer1-SamplerWithoutReplacement-None]` 🆕 | 5.3% (4/76) | 4 | 0.08 | 2026-01-30 |
| `..._replaybuffer[writer1-SamplerWithoutReplacement-storage1]` 🆕 | 5.3% (4/76) | 4 | 0.08 | 2026-01-30 |


### Newly Flaky Tests

- `test/test_rb.py::TestLazyMemmapStorageCleanup::test_cleanup_registry`
- `test/test_distributed.py::TestRayCollector::test_ray_replaybuffer[None-None-None]`
- `test/test_distributed.py::TestRayCollector::test_ray_replaybuffer[None-None-storage1]`
- `test/test_distributed.py::TestRayCollector::test_ray_replaybuffer[None-sampler1-None]`
- `test/test_distributed.py::TestRayCollector::test_ray_replaybuffer[None-sampler1-storage1]`
- `test/test_distributed.py::TestRayCollector::test_ray_replaybuffer[None-SamplerWithoutReplacement-None]`
- `test/test_distributed.py::TestRayCollector::test_ray_replaybuffer[None-SamplerWithoutReplacement-storage1]`
- `test/test_distributed.py::TestRayCollector::test_ray_replaybuffer[writer1-None-None]`
- `test/test_distributed.py::TestRayCollector::test_ray_replaybuffer[writer1-None-storage1]`
- `test/test_distributed.py::TestRayCollector::test_ray_replaybuffer[writer1-sampler1-None]`
- `test/test_distributed.py::TestRayCollector::test_ray_replaybuffer[writer1-sampler1-storage1]`
- `test/test_distributed.py::TestRayCollector::test_ray_replaybuffer[writer1-SamplerWithoutReplacement-None]`
- `test/test_distributed.py::TestRayCollector::test_ray_replaybuffer[writer1-SamplerWithoutReplacement-storage1]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-02-05T06:27:26.738701+00:00*