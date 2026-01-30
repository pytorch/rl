# Flaky Test Report - 2026-01-30

## Summary

- **Flaky tests**: 18
- **Newly flaky** (last 7 days): 18
- **Resolved**: 0
- **Total tests analyzed**: 26761
- **CI runs analyzed**: 150

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...b.py::TestLazyMemmapStorageCleanup::test_cleanup_registry` 🆕 | 16.7% (10/60) | 10 | 0.33 | 2026-01-29 |
| `...nvs.py::TestMultiKeyEnvs::test_rollout[2-5-batch_size0-0]` 🆕 | 6.7% (4/60) | 4 | 0.11 | 2026-01-29 |
| `...nvs.py::TestMultiKeyEnvs::test_rollout[2-5-batch_size1-0]` 🆕 | 6.7% (4/60) | 4 | 0.11 | 2026-01-29 |
| `...nvs.py::TestMultiKeyEnvs::test_rollout[2-5-batch_size2-0]` 🆕 | 6.7% (4/60) | 4 | 0.11 | 2026-01-29 |
| `...test_modules.py::TestMultiAgent::test_multiagent_cnn_lazy` 🆕 | 6.7% (4/60) | 4 | 0.11 | 2026-01-29 |
| `...t_objectives.py::TestSAC::test_sac_prioritized_weights[2]` 🆕 | 6.7% (4/60) | 4 | 0.11 | 2026-01-29 |
| `...::TestRayCollector::test_ray_replaybuffer[None-None-None]` 🆕 | 8.3% (2/24) | 2 | 0.07 | 2026-01-29 |
| `...stRayCollector::test_ray_replaybuffer[None-None-storage1]` 🆕 | 8.3% (2/24) | 2 | 0.07 | 2026-01-29 |
| `...stRayCollector::test_ray_replaybuffer[None-sampler1-None]` 🆕 | 8.3% (2/24) | 2 | 0.07 | 2026-01-29 |
| `...yCollector::test_ray_replaybuffer[None-sampler1-storage1]` 🆕 | 8.3% (2/24) | 2 | 0.07 | 2026-01-29 |
| `...est_ray_replaybuffer[None-SamplerWithoutReplacement-None]` 🆕 | 8.3% (2/24) | 2 | 0.07 | 2026-01-29 |
| `...ray_replaybuffer[None-SamplerWithoutReplacement-storage1]` 🆕 | 8.3% (2/24) | 2 | 0.07 | 2026-01-29 |
| `...estRayCollector::test_ray_replaybuffer[writer1-None-None]` 🆕 | 8.3% (2/24) | 2 | 0.07 | 2026-01-29 |
| `...ayCollector::test_ray_replaybuffer[writer1-None-storage1]` 🆕 | 8.3% (2/24) | 2 | 0.07 | 2026-01-29 |
| `...ayCollector::test_ray_replaybuffer[writer1-sampler1-None]` 🆕 | 8.3% (2/24) | 2 | 0.07 | 2026-01-29 |
| `...llector::test_ray_replaybuffer[writer1-sampler1-storage1]` 🆕 | 8.3% (2/24) | 2 | 0.07 | 2026-01-29 |
| `..._ray_replaybuffer[writer1-SamplerWithoutReplacement-None]` 🆕 | 8.3% (2/24) | 2 | 0.07 | 2026-01-29 |
| `..._replaybuffer[writer1-SamplerWithoutReplacement-storage1]` 🆕 | 8.3% (2/24) | 2 | 0.07 | 2026-01-29 |


### Newly Flaky Tests

- `test/test_rb.py::TestLazyMemmapStorageCleanup::test_cleanup_registry`
- `test/test_envs.py::TestMultiKeyEnvs::test_rollout[2-5-batch_size0-0]`
- `test/test_envs.py::TestMultiKeyEnvs::test_rollout[2-5-batch_size1-0]`
- `test/test_envs.py::TestMultiKeyEnvs::test_rollout[2-5-batch_size2-0]`
- `test/test_modules.py::TestMultiAgent::test_multiagent_cnn_lazy`
- `test/test_objectives.py::TestSAC::test_sac_prioritized_weights[2]`
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

*Generated at 2026-01-30T06:20:36.077661+00:00*