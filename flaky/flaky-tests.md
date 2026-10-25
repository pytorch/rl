# Flaky Test Report - 2026-10-25

## Summary

- **Flaky tests**: 5736
- **Newly flaky** (last 7 days): 2
- **Resolved**: 0
- **Total tests analyzed**: 32760
- **CI runs analyzed**: 60

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...ce[tensor-LazyMemmapStorage-device_data0-device_storage0]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-24 |
| `...torage_device[tensor-LazyMemmapStorage-device_data3-auto]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-24 |
| `...ce[tensor-LazyTensorStorage-device_data0-device_storage0]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-24 |
| `...torage_device[tensor-LazyTensorStorage-device_data3-auto]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-24 |
| `...device[tc-LazyMemmapStorage-device_data0-device_storage0]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-24 |
| `...st_storage_device[tc-LazyMemmapStorage-device_data3-auto]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-24 |
| `...device[tc-LazyTensorStorage-device_data0-device_storage0]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-24 |
| `...st_storage_device[tc-LazyTensorStorage-device_data3-auto]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-24 |
| `...device[td-LazyMemmapStorage-device_data0-device_storage0]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-24 |
| `...st_storage_device[td-LazyMemmapStorage-device_data3-auto]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-24 |
| `...device[td-LazyTensorStorage-device_data0-device_storage0]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-24 |
| `...st_storage_device[td-LazyTensorStorage-device_data3-auto]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-24 |
| `...prioritized_memmap_cuda_sampler_after_multiprocess_writes` | 20.0% (15/75) | 15 | 0.40 | 2026-10-24 |
| `...StorageExport::test_device_storage[cuda-ListStorage-True]` 🆕 | 20.0% (5/25) | 5 | 0.40 | 2026-10-24 |
| `...erGeneration::test_generation_cuda_data_into_cuda_storage` | 20.0% (15/75) | 15 | 0.40 | 2026-10-24 |
| `test/test_checkpoint.py::test_cuda_map_location_and_rng` | 20.0% (15/75) | 15 | 0.40 | 2026-10-24 |
| `...custom_envs.py::TestCustomEnvs::test_financial_env_device` | 20.0% (15/75) | 15 | 0.40 | 2026-10-24 |
| `...st_trainer.py::test_fql_update_matches_recipe[False-cuda]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-24 |
| `...est_trainer.py::test_fql_update_matches_recipe[True-cuda]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-24 |
| `...ict_prioritized_replay_buffer_memmap_storage_cuda_sampler` | 20.0% (15/75) | 15 | 0.40 | 2026-10-24 |


### Newly Flaky Tests

- `test/rb/test_storages.py::TestStorageExport::test_device_storage[cuda-ListStorage-True]`
- `test/objectives/test_transitive_q.py::TestTransitiveQLoss::test_compile_matches_eager`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-10-25T06:28:28.911271+00:00*