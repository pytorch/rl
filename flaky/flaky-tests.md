# Flaky Test Report - 2026-01-31

## Summary

- **Flaky tests**: 13
- **Newly flaky** (last 7 days): 13
- **Resolved**: 0
- **Total tests analyzed**: 26770
- **CI runs analyzed**: 152

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...b.py::TestLazyMemmapStorageCleanup::test_cleanup_registry` 🆕 | 10.0% (10/100) | 10 | 0.20 | 2026-01-29 |
| `...::TestRayCollector::test_ray_replaybuffer[None-None-None]` 🆕 | 10.0% (4/40) | 4 | 0.16 | 2026-01-30 |
| `...stRayCollector::test_ray_replaybuffer[None-None-storage1]` 🆕 | 10.0% (4/40) | 4 | 0.16 | 2026-01-30 |
| `...stRayCollector::test_ray_replaybuffer[None-sampler1-None]` 🆕 | 10.0% (4/40) | 4 | 0.16 | 2026-01-30 |
| `...yCollector::test_ray_replaybuffer[None-sampler1-storage1]` 🆕 | 10.0% (4/40) | 4 | 0.16 | 2026-01-30 |
| `...est_ray_replaybuffer[None-SamplerWithoutReplacement-None]` 🆕 | 10.0% (4/40) | 4 | 0.16 | 2026-01-30 |
| `...ray_replaybuffer[None-SamplerWithoutReplacement-storage1]` 🆕 | 10.0% (4/40) | 4 | 0.16 | 2026-01-30 |
| `...estRayCollector::test_ray_replaybuffer[writer1-None-None]` 🆕 | 10.0% (4/40) | 4 | 0.16 | 2026-01-30 |
| `...ayCollector::test_ray_replaybuffer[writer1-None-storage1]` 🆕 | 10.0% (4/40) | 4 | 0.16 | 2026-01-30 |
| `...ayCollector::test_ray_replaybuffer[writer1-sampler1-None]` 🆕 | 10.0% (4/40) | 4 | 0.16 | 2026-01-30 |
| `...llector::test_ray_replaybuffer[writer1-sampler1-storage1]` 🆕 | 10.0% (4/40) | 4 | 0.16 | 2026-01-30 |
| `..._ray_replaybuffer[writer1-SamplerWithoutReplacement-None]` 🆕 | 10.0% (4/40) | 4 | 0.16 | 2026-01-30 |
| `..._replaybuffer[writer1-SamplerWithoutReplacement-storage1]` 🆕 | 10.0% (4/40) | 4 | 0.16 | 2026-01-30 |


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

*Generated at 2026-01-31T06:15:29.226927+00:00*