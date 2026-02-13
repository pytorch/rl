# Flaky Test Report - 2026-02-13

## Summary

- **Flaky tests**: 12
- **Newly flaky** (last 7 days): 12
- **Resolved**: 0
- **Total tests analyzed**: 26797
- **CI runs analyzed**: 180

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...::TestRayCollector::test_ray_replaybuffer[None-None-None]` 🆕 | 12.0% (12/100) | 12 | 0.24 | 2026-02-13 |
| `...stRayCollector::test_ray_replaybuffer[None-None-storage1]` 🆕 | 12.0% (12/100) | 12 | 0.24 | 2026-02-13 |
| `...stRayCollector::test_ray_replaybuffer[None-sampler1-None]` 🆕 | 12.0% (12/100) | 12 | 0.24 | 2026-02-13 |
| `...yCollector::test_ray_replaybuffer[None-sampler1-storage1]` 🆕 | 12.0% (12/100) | 12 | 0.24 | 2026-02-13 |
| `...est_ray_replaybuffer[None-SamplerWithoutReplacement-None]` 🆕 | 12.0% (12/100) | 12 | 0.24 | 2026-02-13 |
| `...ray_replaybuffer[None-SamplerWithoutReplacement-storage1]` 🆕 | 12.0% (12/100) | 12 | 0.24 | 2026-02-13 |
| `...estRayCollector::test_ray_replaybuffer[writer1-None-None]` 🆕 | 12.0% (12/100) | 12 | 0.24 | 2026-02-13 |
| `...ayCollector::test_ray_replaybuffer[writer1-None-storage1]` 🆕 | 12.0% (12/100) | 12 | 0.24 | 2026-02-13 |
| `...ayCollector::test_ray_replaybuffer[writer1-sampler1-None]` 🆕 | 12.0% (12/100) | 12 | 0.24 | 2026-02-13 |
| `...llector::test_ray_replaybuffer[writer1-sampler1-storage1]` 🆕 | 12.0% (12/100) | 12 | 0.24 | 2026-02-13 |
| `..._ray_replaybuffer[writer1-SamplerWithoutReplacement-None]` 🆕 | 12.0% (12/100) | 12 | 0.24 | 2026-02-13 |
| `..._replaybuffer[writer1-SamplerWithoutReplacement-storage1]` 🆕 | 12.0% (12/100) | 12 | 0.24 | 2026-02-13 |


### Newly Flaky Tests

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

*Generated at 2026-02-13T06:31:03.341185+00:00*