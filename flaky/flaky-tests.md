# Flaky Test Report - 2026-02-06

## Summary

- **Flaky tests**: 13
- **Newly flaky** (last 7 days): 0
- **Resolved**: 0
- **Total tests analyzed**: 26783
- **CI runs analyzed**: 161

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...b.py::TestLazyMemmapStorageCleanup::test_cleanup_registry` | 5.3% (10/190) | 10 | 0.11 | 2026-01-29 |
| `...::TestRayCollector::test_ray_replaybuffer[None-None-None]` | 5.3% (4/76) | 4 | 0.08 | 2026-01-30 |
| `...stRayCollector::test_ray_replaybuffer[None-None-storage1]` | 5.3% (4/76) | 4 | 0.08 | 2026-01-30 |
| `...stRayCollector::test_ray_replaybuffer[None-sampler1-None]` | 5.3% (4/76) | 4 | 0.08 | 2026-01-30 |
| `...yCollector::test_ray_replaybuffer[None-sampler1-storage1]` | 5.3% (4/76) | 4 | 0.08 | 2026-01-30 |
| `...est_ray_replaybuffer[None-SamplerWithoutReplacement-None]` | 5.3% (4/76) | 4 | 0.08 | 2026-01-30 |
| `...ray_replaybuffer[None-SamplerWithoutReplacement-storage1]` | 5.3% (4/76) | 4 | 0.08 | 2026-01-30 |
| `...estRayCollector::test_ray_replaybuffer[writer1-None-None]` | 5.3% (4/76) | 4 | 0.08 | 2026-01-30 |
| `...ayCollector::test_ray_replaybuffer[writer1-None-storage1]` | 5.3% (4/76) | 4 | 0.08 | 2026-01-30 |
| `...ayCollector::test_ray_replaybuffer[writer1-sampler1-None]` | 5.3% (4/76) | 4 | 0.08 | 2026-01-30 |
| `...llector::test_ray_replaybuffer[writer1-sampler1-storage1]` | 5.3% (4/76) | 4 | 0.08 | 2026-01-30 |
| `..._ray_replaybuffer[writer1-SamplerWithoutReplacement-None]` | 5.3% (4/76) | 4 | 0.08 | 2026-01-30 |
| `..._replaybuffer[writer1-SamplerWithoutReplacement-storage1]` | 5.3% (4/76) | 4 | 0.08 | 2026-01-30 |


---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-02-06T06:24:32.758124+00:00*